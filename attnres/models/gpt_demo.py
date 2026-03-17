"""
Minimal GPT-style language model with Block AttnRes residuals.

Designed for:
- Scratch training on small datasets to demonstrate AttnRes scaling benefits
- Comparison against a vanilla GPT baseline (set use_attnres=False)
- Generating scaling-law data (scripts/train_gpt_demo.py)

Architecture:
    Token embedding → N × (CausalSelfAttention → MLP) → LM head
    with Block AttnRes replacing standard residual connections

Reference: Section 5.1 (Scaling Laws) of arXiv:2603.15031
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.block_attn_res import BlockAttnRes
from ..core.utils import RMSNorm


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024       # max sequence length
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    # AttnRes options
    use_attnres: bool = True
    num_attnres_blocks: int = 8   # N in Block AttnRes
    attnres_variant: str = "block"  # "block" | "full"


# -----------------------------------------------------------------------
# Sub-modules
# -----------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Pre-norm LayerNorm with optional bias."""
    def __init__(self, ndim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(self.act(self.c_fc(x))))


class GPTBlock(nn.Module):
    """Single transformer block: attention + MLP, each treated as one layer for AttnRes."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward_split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (attn_out, mlp_out, final_h) for AttnRes hook.

        In AttnRes, each sub-layer (attn, mlp) is a separate layer.
        We expose their outputs so BlockAttnRes can be applied between them.
        """
        attn_out = self.attn(self.ln_1(x))
        h_mid = x + attn_out          # standard residual after attn
        mlp_out = self.mlp(self.ln_2(h_mid))
        h_out = h_mid + mlp_out       # standard residual after mlp
        return attn_out, mlp_out, h_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# -----------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------

class GPTWithAttnRes(nn.Module):
    """GPT-style language model with optional Block AttnRes residuals.

    When use_attnres=True, standard residuals inside each transformer block
    are replaced by Block AttnRes. Each attention sub-layer and each MLP
    sub-layer counts as one AttnRes layer (so n_layer * 2 total layers).

    When use_attnres=False, behaves as a standard GPT baseline.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layer)]),
            ln_f=LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        if cfg.use_attnres:
            # Each block has 2 sub-layers (attn + mlp) → total 2*n_layer AttnRes layers
            n_attnres_layers = cfg.n_layer * 2
            if cfg.attnres_variant == "block":
                self.attnres = BlockAttnRes(
                    num_layers=n_attnres_layers,
                    hidden_dim=cfg.n_embd,
                    num_blocks=cfg.num_attnres_blocks,
                )
            else:
                from ..core.full_attn_res import FullAttnRes
                self.attnres = FullAttnRes(
                    num_layers=n_attnres_layers,
                    hidden_dim=cfg.n_embd,
                )
        else:
            self.attnres = None

        # Init weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layer) (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"Sequence length {T} > block_size {self.cfg.block_size}"

        device = idx.device
        pos = torch.arange(T, device=device)

        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)   # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, n_embd)

        if self.attnres is not None:
            x = self._forward_with_attnres(x)
        else:
            for block in self.transformer.h:
                x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def _forward_with_attnres(self, embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass with Block AttnRes replacing standard residuals."""
        ar = self.attnres

        # Collect all sub-layer outputs first for compute_all_inputs
        layer_outputs = []
        x = embedding
        for block in self.transformer.h:
            # Attention sub-layer output
            attn_out = block.attn(block.ln_1(x))
            layer_outputs.append(attn_out)
            x_after_attn = x + attn_out

            # MLP sub-layer output
            mlp_out = block.mlp(block.ln_2(x_after_attn))
            layer_outputs.append(mlp_out)
            x = x_after_attn + mlp_out

        # Now compute AttnRes inputs for all layers
        attnres_inputs = ar.compute_all_inputs(layer_outputs, embedding)

        # Second pass: apply layers with AttnRes-computed inputs
        for i, block in enumerate(self.transformer.h):
            attn_layer_idx = i * 2
            mlp_layer_idx = i * 2 + 1

            # Attention: use AttnRes-computed input instead of standard h
            h_attn_in = attnres_inputs[attn_layer_idx]
            attn_out = block.attn(block.ln_1(h_attn_in))

            # MLP: use AttnRes-computed input
            h_mlp_in = attnres_inputs[mlp_layer_idx]
            mlp_out = block.mlp(block.ln_2(h_mlp_in))

            # Standard residual merge (AttnRes computes the input, not the skip)
            if i == 0:
                x = embedding + attn_out
            else:
                x = attnres_inputs[attn_layer_idx] + attn_out

            x = attnres_inputs[mlp_layer_idx] + mlp_out

        return x

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

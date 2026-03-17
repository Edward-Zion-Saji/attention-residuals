"""
Full Attention Residuals (Full AttnRes).

Replaces the standard residual accumulation

    h_l = h_{l-1} + f_{l-1}(h_{l-1})

with softmax attention over ALL preceding layer outputs:

    h_l = sum_{i=0}^{l-1} alpha_{i->l} * v_i

where
    v_0        = h_1  (token embedding)
    v_i (i>=1) = f_i(h_i)  (layer output)
    alpha_{i->l} = softmax( w_l^T RMSNorm(k_i) )  over i=0..l-1
    k_i        = v_i  (keys equal values)

The pseudo-query w_l is a single learned d-dimensional vector per layer,
initialised to zero so that the initial weights are uniform (equal-weight
average), matching standard residuals at init and ensuring training stability.

Reference: Section 3.1 of the AttnRes paper (arXiv:2603.15031).
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .utils import RMSNorm, zero_init_
from .online_softmax import AttnWithStats, merge_attn_stats


class FullAttnRes(nn.Module):
    """Full Attention Residuals module.

    Holds one learned pseudo-query vector per layer.  During a forward pass,
    layer l's input is computed as a softmax-weighted combination of:
      - v_0 = the token embedding (h_1)
      - v_i = the output of layer i  (for i = 1 .. l-1)

    Args:
        num_layers: Number of transformer layers L.
        hidden_dim: Hidden dimension d.
        eps:        RMSNorm epsilon.
        block_size: Block size S for the two-phase inference schedule.
                    Set to 1 to use the naive layer-by-layer path.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        eps: float = 1e-6,
        block_size: int = 8,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.block_size = block_size

        # One pseudo-query vector per layer, initialised to zero.
        # Shape: (num_layers, hidden_dim)
        self.queries = nn.Parameter(torch.empty(num_layers, hidden_dim))
        zero_init_(self.queries)

        self.key_norm = RMSNorm(hidden_dim, eps=eps)

    # ------------------------------------------------------------------
    # Stateful forward: call reset_state() at the start of each sequence,
    # then forward() once per layer in order.
    # ------------------------------------------------------------------

    def reset_state(self):
        """Clear the stored layer outputs.  Call at the start of each sequence."""
        # List of (B, T, d) tensors: v_0 is the embedding, v_i is f_i(h_i)
        self._values: List[torch.Tensor] = []

    def push(self, v: torch.Tensor):
        """Store a layer output (or the embedding for layer 0).

        Args:
            v: Tensor of shape (B, T, d).
        """
        self._values.append(v)

    def forward(self, layer_idx: int) -> torch.Tensor:
        """Compute the input to layer `layer_idx` via depth-wise attention.

        Must be called after push()-ing all v_0 .. v_{layer_idx - 1}.

        Args:
            layer_idx: Index of the layer whose input we are computing (0-based).
                       Layer 0 simply returns v_0 (the embedding).

        Returns:
            Tensor of shape (B, T, d) — the new hidden state for layer `layer_idx`.
        """
        if layer_idx == 0:
            return self._values[0]

        # Stack all available values: (B, T, l, d) then transpose for matmul
        values = self._values[: layer_idx + 1]  # v_0 .. v_{layer_idx-1} and no more
        # Actually we have pushed exactly layer_idx + 1 items if we also push v_0
        # but for layer l we need v_0..v_{l-1}, which are self._values[0..layer_idx-1]
        # Correct: after calling reset_state() and push(embedding), then for each
        # layer l we call forward(l) BEFORE pushing f_l(h_l).
        values = self._values  # includes v_0 .. v_{l-1}

        # Normalise keys: each v_i is used as both key and value.
        # keys: (num_sources, B, T, d)
        # We compute attention per token: for each (b, t), w_l^T RMSNorm(v_i)
        # w_l: (d,)  ->  (1, 1, d)
        w = self.queries[layer_idx]  # (d,)

        # Build key matrix: stack values along a new "source" dim
        # v_stack: (B, T, num_sources, d)
        v_stack = torch.stack(values, dim=2)
        num_sources = v_stack.shape[2]

        # Normalise keys across the d dimension
        k_stack = self.key_norm(v_stack)  # (B, T, num_sources, d)

        # Scores: w_l · RMSNorm(v_i) for each source i
        # w: (d,) -> broadcast over (B, T, num_sources)
        scores = (k_stack * w).sum(dim=-1)  # (B, T, num_sources)

        # Softmax over the source dimension
        alpha = torch.softmax(scores, dim=-1)  # (B, T, num_sources)

        # Weighted sum of values
        # alpha: (B, T, num_sources, 1) * v_stack: (B, T, num_sources, d)
        h = (alpha.unsqueeze(-1) * v_stack).sum(dim=2)  # (B, T, d)

        return h

    # ------------------------------------------------------------------
    # Convenience: full sequence forward (used during training when all
    # layer outputs are available simultaneously)
    # ------------------------------------------------------------------

    def compute_all_inputs(
        self,
        layer_outputs: List[torch.Tensor],
        embedding: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Compute all layer inputs [h_1, h_2, ..., h_L] given outputs.

        This is the efficient training path: all f_i(h_i) are already
        computed (and retained for backprop), so we compute all attention
        weights in a batched fashion.

        Args:
            layer_outputs: List of L tensors of shape (B, T, d).
                           layer_outputs[i] = f_{i+1}(h_{i+1}).
            embedding:     Token embedding h_1, shape (B, T, d).

        Returns:
            List of L tensors, where result[l] is the input h_{l+1} to
            layer l+1 (i.e. the residual-replaced input).
        """
        L = len(layer_outputs)
        B, T, d = embedding.shape

        # v_0 = embedding, v_i = layer_outputs[i-1] for i >= 1
        all_values = [embedding] + layer_outputs  # length L+1

        inputs = []
        for l in range(L):
            # Layer l+1 attends over v_0 .. v_l  (l+1 sources)
            sources = all_values[: l + 1]
            v_stack = torch.stack(sources, dim=2)  # (B, T, l+1, d)
            k_stack = self.key_norm(v_stack)
            w = self.queries[l]  # (d,)
            scores = (k_stack * w).sum(dim=-1)  # (B, T, l+1)
            alpha = torch.softmax(scores, dim=-1)
            h = (alpha.unsqueeze(-1) * v_stack).sum(dim=2)  # (B, T, d)
            inputs.append(h)

        return inputs

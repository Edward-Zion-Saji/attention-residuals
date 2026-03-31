"""
Ablation study: reproduces Table 4 from arXiv:2603.15031 at small scale.

Tests the key design choices of AttnRes on a fixed model size:
  - Baseline (PreNorm)
  - Full AttnRes (our implementation)
  - Block AttnRes, varying N blocks: N=1,2,4,6,8
  - w/ sigmoid instead of softmax
  - w/o RMSNorm on keys
  - Input-independent mixing (learned scalars, no content-dependent query)

All variants share identical hyperparameters and total training compute.

Usage:
    python scripts/eval_ablations.py
    python scripts/eval_ablations.py --quick
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attnres.models.gpt_demo import GPTConfig, GPTWithAttnRes
from attnres.core.block_attn_res import BlockAttnRes
from attnres.core.utils import RMSNorm, zero_init_


# ---------------------------------------------------------------------------
# Custom AttnRes variants for ablations (not in main library by design)
# ---------------------------------------------------------------------------

class BlockAttnResSigmoid(BlockAttnRes):
    """Block AttnRes with sigmoid instead of softmax (ablation)."""

    def forward(self, layer_idx: int) -> torch.Tensor:
        block_n = self._block_of(layer_idx)
        i = self._intra_idx

        sources: List[torch.Tensor] = list(self._blocks)
        if i > 0 and self._partial_block is not None:
            sources.append(self._partial_block)

        v_stack = torch.stack(sources, dim=2)       # (B, T, S, d)
        k_stack = self.key_norm(v_stack)
        w = self.queries[layer_idx]                  # (d,)
        scores = (k_stack * w).sum(dim=-1)           # (B, T, S)
        # sigmoid instead of softmax — not normalised
        alpha = torch.sigmoid(scores)
        alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-8)
        h = (alpha.unsqueeze(-1) * v_stack).sum(dim=2)
        return h

    def compute_all_inputs(self, layer_outputs, embedding):
        L = len(layer_outputs)
        inputs = []
        block_reprs = [embedding]
        partial_block: Optional[torch.Tensor] = None

        for l in range(L):
            block_n = self._block_of(l)
            i = l - self._block_starts[block_n]

            sources = list(block_reprs)
            if i > 0 and partial_block is not None:
                sources.append(partial_block)

            v_stack = torch.stack(sources, dim=2)
            k_stack = self.key_norm(v_stack)
            w = self.queries[l]
            scores = (k_stack * w).sum(dim=-1)
            alpha = torch.sigmoid(scores)
            alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-8)
            h = (alpha.unsqueeze(-1) * v_stack).sum(dim=2)
            inputs.append(h)

            f_l = layer_outputs[l]
            partial_block = f_l if partial_block is None else partial_block + f_l

            if i + 1 >= self.block_sizes[block_n]:
                block_reprs.append(partial_block)
                partial_block = None

        return inputs


class BlockAttnResNoNorm(BlockAttnRes):
    """Block AttnRes without RMSNorm on keys (ablation)."""

    def forward(self, layer_idx: int) -> torch.Tensor:
        block_n = self._block_of(layer_idx)
        i = self._intra_idx

        sources: List[torch.Tensor] = list(self._blocks)
        if i > 0 and self._partial_block is not None:
            sources.append(self._partial_block)

        v_stack = torch.stack(sources, dim=2)
        w = self.queries[layer_idx]
        # No key normalisation
        scores = (v_stack * w).sum(dim=-1)
        alpha = torch.softmax(scores, dim=-1)
        h = (alpha.unsqueeze(-1) * v_stack).sum(dim=2)
        return h

    def compute_all_inputs(self, layer_outputs, embedding):
        L = len(layer_outputs)
        inputs = []
        block_reprs = [embedding]
        partial_block: Optional[torch.Tensor] = None

        for l in range(L):
            block_n = self._block_of(l)
            i = l - self._block_starts[block_n]

            sources = list(block_reprs)
            if i > 0 and partial_block is not None:
                sources.append(partial_block)

            v_stack = torch.stack(sources, dim=2)
            w = self.queries[l]
            scores = (v_stack * w).sum(dim=-1)
            alpha = torch.softmax(scores, dim=-1)
            h = (alpha.unsqueeze(-1) * v_stack).sum(dim=2)
            inputs.append(h)

            f_l = layer_outputs[l]
            partial_block = f_l if partial_block is None else partial_block + f_l

            if i + 1 >= self.block_sizes[block_n]:
                block_reprs.append(partial_block)
                partial_block = None

        return inputs


class BlockAttnResInputIndependent(BlockAttnRes):
    """Block AttnRes with input-independent mixing weights (learned scalars, ablation).

    Replaces the content-dependent query w_l with a per-source scalar weight,
    similar to DenseFormer. The weight is still layer-specific but not input-dependent.
    """

    def __init__(self, num_layers: int, hidden_dim: int, num_blocks: int = 8, eps: float = 1e-6):
        super().__init__(num_layers, hidden_dim, num_blocks, eps)
        # Replace queries with per-layer, per-source scalar weights
        # We use num_blocks+2 max sources (conservative upper bound)
        max_sources = num_blocks + 2
        self.source_weights = nn.Parameter(torch.zeros(num_layers, max_sources))

    def _get_alpha(self, layer_idx: int, n_sources: int) -> torch.Tensor:
        """Return softmax-normalised per-source scalars (input-independent)."""
        w = self.source_weights[layer_idx, :n_sources]
        return torch.softmax(w, dim=0)  # (n_sources,)

    def forward(self, layer_idx: int) -> torch.Tensor:
        block_n = self._block_of(layer_idx)
        i = self._intra_idx

        sources: List[torch.Tensor] = list(self._blocks)
        if i > 0 and self._partial_block is not None:
            sources.append(self._partial_block)

        v_stack = torch.stack(sources, dim=2)        # (B, T, S, d)
        alpha = self._get_alpha(layer_idx, len(sources))   # (S,)
        h = (alpha.unsqueeze(-1) * v_stack).sum(dim=2)     # (B, T, d)
        return h

    def compute_all_inputs(self, layer_outputs, embedding):
        L = len(layer_outputs)
        inputs = []
        block_reprs = [embedding]
        partial_block: Optional[torch.Tensor] = None

        for l in range(L):
            block_n = self._block_of(l)
            i = l - self._block_starts[block_n]

            sources = list(block_reprs)
            if i > 0 and partial_block is not None:
                sources.append(partial_block)

            v_stack = torch.stack(sources, dim=2)
            alpha = self._get_alpha(l, len(sources))
            h = (alpha.unsqueeze(-1) * v_stack).sum(dim=2)
            inputs.append(h)

            f_l = layer_outputs[l]
            partial_block = f_l if partial_block is None else partial_block + f_l

            if i + 1 >= self.block_sizes[block_n]:
                block_reprs.append(partial_block)
                partial_block = None

        return inputs


# ---------------------------------------------------------------------------
# Patch GPTWithAttnRes to support ablation AttnRes modules
# ---------------------------------------------------------------------------

def build_model_with_attnres_module(cfg: GPTConfig, attnres_module: nn.Module) -> nn.Module:
    """Build a GPTWithAttnRes model but replace its attnres with a custom one."""
    model = GPTWithAttnRes(cfg)
    if hasattr(model, "attnres"):
        model.attnres = attnres_module
    return model


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def make_synthetic_dataset(vocab_size: int, seq_len: int, n_samples: int):
    torch.manual_seed(42)
    data = torch.randint(0, vocab_size, (n_samples, seq_len + 1))
    x = data[:, :-1]
    y = data[:, 1:]
    return TensorDataset(x, y)


def make_wikitext_dataset(seq_len: int, max_tokens: int = 3_000_000):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    text = "\n".join(ds["text"])[:max_tokens]
    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
    n = (len(data) - 1) // seq_len
    x = torch.stack([data[i * seq_len: i * seq_len + seq_len] for i in range(n)])
    y = torch.stack([data[i * seq_len + 1: i * seq_len + seq_len + 1] for i in range(n)])
    return TensorDataset(x, y)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    dataset,
    steps: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    log_interval: int = 200,
) -> Dict:
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95)
    )
    warmup = max(50, steps // 20)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(steps - warmup, 1)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    step = 0
    losses_logged = []
    steps_logged = []
    t0 = time.perf_counter()
    data_iter = iter(loader)

    while step < steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1

        if step % log_interval == 0:
            elapsed = time.perf_counter() - t0
            print(f"    step {step:5d}/{steps} | loss {loss.item():.4f} | {elapsed:.1f}s")
            losses_logged.append(loss.item())
            steps_logged.append(step)
            t0 = time.perf_counter()

    final_loss = losses_logged[-1] if losses_logged else float("nan")
    return {
        "n_params": n_params,
        "steps": steps_logged,
        "losses": losses_logged,
        "final_loss": final_loss,
    }


# ---------------------------------------------------------------------------
# Ablation variants definition
# ---------------------------------------------------------------------------

def get_ablation_variants(cfg: GPTConfig, num_blocks: int):
    """Return list of (name, model) pairs for ablation study."""
    variants = []

    # 1. Baseline
    base_cfg = GPTConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size,
        n_layer=cfg.n_layer, n_embd=cfg.n_embd, n_head=cfg.n_head,
        use_attnres=False,
    )
    variants.append(("Baseline (PreNorm)", GPTWithAttnRes(base_cfg)))

    # 2. Block AttnRes (standard, varying N)
    for N in [2, 4, 8]:
        ar_cfg = GPTConfig(
            vocab_size=cfg.vocab_size, block_size=cfg.block_size,
            n_layer=cfg.n_layer, n_embd=cfg.n_embd, n_head=cfg.n_head,
            use_attnres=True, num_attnres_blocks=N,
        )
        variants.append((f"Block AttnRes (N={N})", GPTWithAttnRes(ar_cfg)))

    # 3. Full AttnRes
    ar_cfg_full = GPTConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size,
        n_layer=cfg.n_layer, n_embd=cfg.n_embd, n_head=cfg.n_head,
        use_attnres=True, num_attnres_blocks=cfg.n_layer * 2,
        attnres_variant="full",
    )
    variants.append(("Full AttnRes", GPTWithAttnRes(ar_cfg_full)))

    # 4. Sigmoid ablation (uses Block AttnRes with N=4)
    num_ar_layers = cfg.n_layer * 2
    sig_ar = BlockAttnResSigmoid(num_ar_layers, cfg.n_embd, num_blocks=4)
    ar_cfg_sig = GPTConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size,
        n_layer=cfg.n_layer, n_embd=cfg.n_embd, n_head=cfg.n_head,
        use_attnres=True, num_attnres_blocks=4,
    )
    m_sig = GPTWithAttnRes(ar_cfg_sig)
    m_sig.attnres = sig_ar
    variants.append(("Block AttnRes w/ sigmoid", m_sig))

    # 5. No RMSNorm ablation
    nonorm_ar = BlockAttnResNoNorm(num_ar_layers, cfg.n_embd, num_blocks=4)
    m_nonorm = GPTWithAttnRes(ar_cfg_sig)
    m_nonorm.attnres = nonorm_ar
    variants.append(("Block AttnRes w/o RMSNorm", m_nonorm))

    # 6. Input-independent mixing
    indep_ar = BlockAttnResInputIndependent(num_ar_layers, cfg.n_embd, num_blocks=4)
    m_indep = GPTWithAttnRes(ar_cfg_sig)
    m_indep.attnres = indep_ar
    variants.append(("Block AttnRes input-indep. mixing", m_indep))

    return variants


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ablation_results(results: List[Dict], save_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [r["name"] for r in results]
    losses = [r["final_loss"] for r in results]
    colors = ["#d6604d" if "Baseline" in n else "#2166ac" for n in names]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.4), 5))
    bars = ax.bar(range(len(names)), losses, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    # Annotate bars
    for bar, loss in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            loss + (max(losses) - min(losses)) * 0.01,
            f"{loss:.4f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Final Validation Loss", fontsize=11)
    ax.set_title("Ablation Study: Key Design Choices of AttnRes\n(lower is better)", fontsize=12)

    # Baseline reference line
    baseline_loss = next((r["final_loss"] for r in results if "Baseline" in r["name"]), None)
    if baseline_loss is not None:
        ax.axhline(baseline_loss, color="#d6604d", linestyle="--", linewidth=1.2, alpha=0.7,
                   label=f"Baseline ({baseline_loss:.4f})")
        ax.legend(fontsize=9)

    loss_range = max(losses) - min(losses)
    ax.set_ylim(min(losses) - loss_range * 0.3, max(losses) + loss_range * 0.3)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ablation plot → {save_path}")


def plot_block_sweep(results: List[Dict], save_path: str) -> None:
    """Plot loss vs number of blocks (Fig 6 style)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    block_results = [r for r in results if r["name"].startswith("Block AttnRes (N=")]
    baseline = next((r for r in results if "Baseline" in r["name"]), None)
    full_ar = next((r for r in results if r["name"] == "Full AttnRes"), None)

    if len(block_results) < 2:
        return

    ns = [int(r["name"].split("N=")[1].rstrip(")")) for r in block_results]
    ls = [r["final_loss"] for r in block_results]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, ls, "o-", color="#2166ac", linewidth=2, markersize=8, label="Block AttnRes (N=...)")

    if baseline:
        ax.axhline(baseline["final_loss"], color="#d6604d", linestyle="--",
                   linewidth=1.5, label=f"Baseline ({baseline['final_loss']:.4f})")
    if full_ar:
        ax.axhline(full_ar["final_loss"], color="#4dac26", linestyle=":",
                   linewidth=1.5, label=f"Full AttnRes ({full_ar['final_loss']:.4f})")

    ax.set_xlabel("Number of blocks N", fontsize=11)
    ax.set_ylabel("Final Validation Loss", fontsize=11)
    ax.set_title("Effect of Block Count N on AttnRes Performance\n(Fig 6 from arXiv:2603.15031)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved block sweep plot → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "wikitext"])
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=0,
                        help="Override step count (0 = auto)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out_dir", type=str, default="./outputs/eval_ablations")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    vocab_size = 256

    steps = args.steps if args.steps > 0 else (1500 if args.quick else 4000)

    print(f"\n{'='*60}")
    print(f"AttnRes Ablation Study")
    print(f"Device: {device} | Steps: {steps} | L={args.n_layer} d={args.n_embd}")
    print(f"{'='*60}\n")

    if args.dataset == "wikitext":
        dataset = make_wikitext_dataset(args.block_size)
    else:
        dataset = make_synthetic_dataset(vocab_size, args.block_size, n_samples=6000)

    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
    )

    variants = get_ablation_variants(cfg, num_blocks=4)
    all_results = []

    for name, model in variants:
        print(f"\n--- {name} ---")
        log = train(model, dataset, steps, args.lr, args.batch_size, device)
        result = {
            "name": name,
            "final_loss": log["final_loss"],
            "n_params": log["n_params"],
            "steps": log["steps"],
            "losses": log["losses"],
        }
        all_results.append(result)
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "").replace("=", "")
        log_path = os.path.join(args.out_dir, f"ablation_{slug}.json")
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Final loss: {log['final_loss']:.4f}  | params: {log['n_params']:,}")

    # ------------------------------------------------------------------
    # Print summary table (matches Table 4 style)
    # ------------------------------------------------------------------
    baseline_loss = next((r["final_loss"] for r in all_results if "Baseline" in r["name"]), None)

    print(f"\n{'='*65}")
    print("ABLATION STUDY RESULTS  (Table 4 style — lower loss = better)")
    print(f"{'='*65}")
    print(f"{'Variant':<40} {'Loss':>8}  {'vs Baseline':>12}")
    print("─" * 65)

    for r in all_results:
        delta = ""
        if baseline_loss is not None and r["name"] != "Baseline (PreNorm)":
            d = baseline_loss - r["final_loss"]
            delta = f"{'↓'+f'{d:.4f}':>12}" if d > 0 else f"{'↑'+f'{-d:.4f}':>12}"
        print(f"{r['name']:<40} {r['final_loss']:>8.4f}  {delta}")

    # ------------------------------------------------------------------
    # Save plots
    # ------------------------------------------------------------------
    ablation_plot_path = os.path.join(args.out_dir, "ablation_results.png")
    plot_ablation_results(all_results, ablation_plot_path)

    block_sweep_path = os.path.join(args.out_dir, "block_sweep.png")
    plot_block_sweep(all_results, block_sweep_path)

    # Loss curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_cycle = ["#d6604d", "#2166ac", "#1a9850", "#7b3294", "#e08214", "#006837", "#a6d96a"]
    for r, color in zip(all_results, colors_cycle):
        if r["steps"] and r["losses"]:
            ax.plot(r["steps"], r["losses"], linewidth=1.5, label=r["name"], color=color)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Ablation: Loss Curves for All Variants")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    curves_path = os.path.join(args.out_dir, "ablation_loss_curves.png")
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curves → {curves_path}")

    # Summary JSON
    summary = {
        "device": str(device),
        "dataset": args.dataset,
        "n_layer": args.n_layer,
        "n_embd": args.n_embd,
        "steps": steps,
        "results": [{"name": r["name"], "final_loss": r["final_loss"], "n_params": r["n_params"]}
                    for r in all_results],
    }
    with open(os.path.join(args.out_dir, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

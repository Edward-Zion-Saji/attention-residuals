"""
Depth-wise attention analysis: reproduces Fig 8 from arXiv:2603.15031.

Trains a model with Block AttnRes and Full AttnRes, then visualises the
learned alpha_{i->l} weight matrices showing which source layers each
consuming layer attends to.

Key observations from the paper:
  - Diagonal dominance: each layer primarily attends to its predecessor
  - Persistent embedding weight: source 0 (embedding) retains non-trivial weight
  - Off-diagonal concentrations: learned skip connections emerge
  - Layer specialisation: pre-attn vs pre-MLP layers show different patterns

Usage:
    python scripts/eval_depth_attention.py
    python scripts/eval_depth_attention.py --quick
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attnres.models.gpt_demo import GPTConfig, GPTWithAttnRes
from attnres.core.block_attn_res import BlockAttnRes
from attnres.core.full_attn_res import FullAttnRes
from attnres.core.utils import RMSNorm
from attnres.visualisation.attention_maps import plot_depth_attention, compute_attention_weights


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_synthetic_dataset(vocab_size, seq_len, n_samples):
    torch.manual_seed(42)
    data = torch.randint(0, vocab_size, (n_samples, seq_len + 1))
    return TensorDataset(data[:, :-1], data[:, 1:])


def make_wikitext_dataset(seq_len, max_tokens=2_000_000):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    text = "\n".join(ds["text"])[:max_tokens]
    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
    n = (len(data) - 1) // seq_len
    x = torch.stack([data[i * seq_len: i * seq_len + seq_len] for i in range(n)])
    y = torch.stack([data[i * seq_len + 1: i * seq_len + seq_len + 1] for i in range(n)])
    return TensorDataset(x, y)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg, dataset, steps, lr, batch_size, device, log_interval=200):
    model = GPTWithAttnRes(cfg).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))

    warmup = max(50, steps // 20)
    def lr_fn(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(steps - warmup, 1)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * p)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    model.train()
    step = 0
    losses = []
    data_iter = iter(loader)
    t0 = time.perf_counter()

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
            print(f"  step {step}/{steps} | loss {loss.item():.4f} | {elapsed:.1f}s")
            losses.append(loss.item())
            t0 = time.perf_counter()

    return model, losses


# ---------------------------------------------------------------------------
# Compute alpha_{i->l} from actual data (not just zero-init random)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_data_driven_weights(model: GPTWithAttnRes, dataset, device, n_batches=10) -> torch.Tensor:
    """Compute alpha_{i->l} averaged over real data samples.

    Instead of using zero-init queries (which give uniform weights at init),
    this uses the trained queries against actual block representations derived
    from real data.

    Returns:
        weights: (n_layers, max_sources) tensor with averaged alpha values.
    """
    attnres = model.attnres
    if not isinstance(attnres, (BlockAttnRes, FullAttnRes)):
        return None

    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    num_layers = attnres.num_layers
    d = attnres.hidden_dim

    if isinstance(attnres, BlockAttnRes):
        max_sources = attnres.num_blocks + 2
    else:
        max_sources = num_layers + 1

    weight_accumulator = torch.zeros(num_layers, max_sources)
    count = 0

    model.eval()
    for batch_idx, (x, _) in enumerate(loader):
        if batch_idx >= n_batches:
            break
        x = x.to(device)

        # Collect actual layer outputs via hooks
        layer_outputs = []
        hooks = []

        if isinstance(attnres, BlockAttnRes):
            # Hook into the forward pass to capture actual f_l outputs
            embedding_captured = []

            def emb_hook(module, input, output):
                # output is (B, T, d)
                if len(embedding_captured) == 0:
                    embedding_captured.append(output.detach().cpu())

            # Hook into the embedding
            emb_h = model.transformer.wte.register_forward_hook(
                lambda m, i, o: embedding_captured.append(o.detach().cpu()) or None
            )

            attn_outs = []
            mlp_outs = []
            for blk in model.transformer.h:
                def make_attn_hook(lst):
                    def hook(m, inp, out):
                        lst.append(out.detach().cpu())
                    return hook
                def make_mlp_hook(lst):
                    def hook(m, inp, out):
                        lst.append(out.detach().cpu())
                    return hook
                hooks.append(blk.attn.register_forward_hook(make_attn_hook(attn_outs)))
                hooks.append(blk.mlp.register_forward_hook(make_mlp_hook(mlp_outs)))

            _ = model(x)
            for h in hooks:
                h.remove()
            emb_h.remove()

            if not embedding_captured:
                continue
            embedding = embedding_captured[-1]  # (B, T, d)

            # Interleave attn/mlp outputs into ordered layer outputs
            ordered = []
            for a, m in zip(attn_outs, mlp_outs):
                ordered.append(a)
                ordered.append(m)

            if len(ordered) != num_layers:
                continue

            # Build block representations
            block_reprs = [embedding]   # b_0
            partial = None
            for l in range(num_layers):
                block_n = attnres._block_of(l)
                i = l - attnres._block_starts[block_n]
                f_l = ordered[l]
                partial = f_l if partial is None else partial + f_l
                if i + 1 >= attnres.block_sizes[block_n]:
                    block_reprs.append(partial)
                    partial = None

            # Compute alpha per layer from real block representations
            for l in range(num_layers):
                block_n = attnres._block_of(l)
                i = l - attnres._block_starts[block_n]
                n_completed = min(block_n + 1, len(block_reprs))
                sources = block_reprs[:n_completed]

                if i > 0:
                    block_start = attnres._block_starts[block_n]
                    partial_l = sum(ordered[block_start + j] for j in range(i))
                    sources = sources + [partial_l]

                v_stack = torch.stack(sources, dim=0)   # (S, B, T, d)
                v_stack = v_stack.permute(1, 2, 0, 3)   # (B, T, S, d)
                k_stack = attnres.key_norm(v_stack.to(device)).cpu()
                w = attnres.queries[l].detach().cpu()

                scores = (k_stack * w).sum(dim=-1)       # (B, T, S)
                alpha = torch.softmax(scores, dim=-1).mean(dim=(0, 1))  # (S,)
                weight_accumulator[l, :len(alpha)] += alpha

        count += 1

    if count > 0:
        weight_accumulator /= count

    return weight_accumulator


# ---------------------------------------------------------------------------
# Heatmap plotting (extended version of attention_maps.py plot)
# ---------------------------------------------------------------------------

def plot_depth_heatmap_extended(
    weights: torch.Tensor,
    n_layer: int,
    n_embd: int,
    num_blocks: int,
    save_path: str,
    title: str = "Depth-wise Attention Weights",
):
    """Plot a detailed heatmap with annotations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    W = weights.numpy() if torch.is_tensor(weights) else weights

    fig, ax = plt.subplots(figsize=(min(16, W.shape[1] + 2), min(12, n_layer + 2)))

    import matplotlib.colors as mcolors
    cmap = plt.cm.Blues
    cmap.set_bad(color="#f0f0f0")

    import numpy as np
    W_masked = np.ma.masked_where(np.isnan(W), W)
    im = ax.imshow(W_masked, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="α_{i→l} (attention weight)", fraction=0.03, pad=0.02)

    ax.set_xlabel("Source index i (0 = embedding, then block reps)", fontsize=11)
    ax.set_ylabel("Consuming layer l", fontsize=11)
    ax.set_title(title, fontsize=12)

    n_rows, n_cols = W.shape
    tick_step_r = max(1, n_rows // 16)
    tick_step_c = max(1, n_cols // 16)
    ax.set_yticks(range(0, n_rows, tick_step_r))
    ax.set_yticklabels([f"L{i}" for i in range(0, n_rows, tick_step_r)], fontsize=7)
    ax.set_xticks(range(0, n_cols, tick_step_c))
    ax.set_xticklabels([f"S{i}" for i in range(0, n_cols, tick_step_c)], fontsize=7)

    # Highlight block boundaries
    if num_blocks > 1:
        block_size = n_rows // num_blocks
        for b in range(1, num_blocks):
            ax.axhline(b * block_size - 0.5, color="red", linewidth=0.8, alpha=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap → {save_path}")


# ---------------------------------------------------------------------------
# Statistics reporting
# ---------------------------------------------------------------------------

def analyse_weight_matrix(weights: torch.Tensor, name: str) -> Dict:
    """Compute statistics about the learned attention patterns."""
    import numpy as np
    W = weights.numpy() if torch.is_tensor(weights) else weights

    n_layers, max_sources = W.shape

    # Diagonal weight (locality): how much does each layer attend to its predecessor?
    diag_weights = []
    for l in range(n_layers):
        # Find the largest index that is not NaN in row l
        valid = ~np.isnan(W[l])
        if valid.any():
            # The "predecessor" in block representation is the last valid source
            last_valid = np.where(valid)[0][-1]
            diag_weights.append(W[l, last_valid])

    # Embedding weight (source 0)
    embedding_weights = W[:, 0]
    embedding_weights = embedding_weights[~np.isnan(embedding_weights)]

    # Entropy of attention distribution (lower = sharper / more selective)
    entropies = []
    for l in range(n_layers):
        row = W[l]
        valid = row[~np.isnan(row)]
        if len(valid) > 1:
            ent = -np.sum(valid * np.log(valid + 1e-9))
            entropies.append(ent)

    stats = {
        "name": name,
        "mean_diagonal_weight": float(np.mean(diag_weights)) if diag_weights else float("nan"),
        "mean_embedding_weight": float(np.mean(embedding_weights)) if len(embedding_weights) else float("nan"),
        "mean_entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "min_entropy": float(np.min(entropies)) if entropies else float("nan"),
        "max_entropy": float(np.max(entropies)) if entropies else float("nan"),
    }
    return stats


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
    parser.add_argument("--num_attnres_blocks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out_dir", type=str, default="./outputs/eval_depth_attention")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    vocab_size = 256
    steps = args.steps if args.steps > 0 else (1500 if args.quick else 5000)

    print(f"\n{'='*60}")
    print(f"AttnRes Depth Attention Analysis")
    print(f"Device: {device} | Steps: {steps} | L={args.n_layer} d={args.n_embd}")
    print(f"{'='*60}\n")

    if args.dataset == "wikitext":
        dataset = make_wikitext_dataset(args.block_size)
    else:
        dataset = make_synthetic_dataset(vocab_size, args.block_size, n_samples=6000)

    all_stats = []

    for variant, use_attnres, attnres_variant in [
        ("Block AttnRes", True, "block"),
        ("Full AttnRes",  True, "full"),
    ]:
        print(f"\n--- Training {variant} ---")
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            n_head=args.n_head,
            use_attnres=use_attnres,
            num_attnres_blocks=args.num_attnres_blocks,
            attnres_variant=attnres_variant,
        )
        model, losses = train(cfg, dataset, steps, args.lr, args.batch_size, device)

        # Save checkpoint
        slug = variant.lower().replace(" ", "_")
        ckpt_path = os.path.join(args.out_dir, f"{slug}_model.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint → {ckpt_path}")

        # 1. Zero-init / structural heatmap (from query params only)
        structural_weights = compute_attention_weights(model.attnres)
        struct_path = os.path.join(args.out_dir, f"{slug}_structural_heatmap.png")
        plot_depth_heatmap_extended(
            structural_weights,
            n_layer=args.n_layer * 2,
            n_embd=args.n_embd,
            num_blocks=args.num_attnres_blocks,
            save_path=struct_path,
            title=f"{variant} — Structural Weights (trained queries, random keys)",
        )

        # 2. Data-driven heatmap (from actual block representations)
        print(f"  Computing data-driven attention weights...")
        data_weights = compute_data_driven_weights(model, dataset, device, n_batches=20)
        if data_weights is not None:
            data_path = os.path.join(args.out_dir, f"{slug}_data_driven_heatmap.png")
            plot_depth_heatmap_extended(
                data_weights,
                n_layer=args.n_layer * 2,
                n_embd=args.n_embd,
                num_blocks=args.num_attnres_blocks,
                save_path=data_path,
                title=f"{variant} — Data-Driven Weights (α_{{i→l}} averaged over samples)",
            )

            # Statistics
            stats = analyse_weight_matrix(data_weights, variant)
            all_stats.append(stats)
            print(f"  Diagonal weight (locality): {stats['mean_diagonal_weight']:.4f}")
            print(f"  Embedding weight:           {stats['mean_embedding_weight']:.4f}")
            print(f"  Mean entropy:               {stats['mean_entropy']:.4f}")

        # Also use built-in plot_depth_attention for the standard view
        std_path = os.path.join(args.out_dir, f"{slug}_depth_attention.png")
        plot_depth_attention(model, save_path=std_path,
                              title=f"Depth-Wise Attention Weights — {variant}")

    # ------------------------------------------------------------------
    # Print summary table of attention statistics
    # ------------------------------------------------------------------
    if all_stats:
        print(f"\n{'='*65}")
        print("DEPTH ATTENTION STATISTICS")
        print(f"{'='*65}")
        print(f"{'Variant':<25} {'Diag. Weight':>14} {'Embed. Weight':>14} {'Entropy':>10}")
        print("─" * 65)
        for s in all_stats:
            print(f"{s['name']:<25} {s['mean_diagonal_weight']:>14.4f} "
                  f"{s['mean_embedding_weight']:>14.4f} {s['mean_entropy']:>10.4f}")

    # Save stats
    with open(os.path.join(args.out_dir, "depth_stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nAll outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

"""
Train a small GPT with and without Block AttnRes and compare loss curves.

Reproduces the scaling law experiment from Fig 4 of arXiv:2603.15031 at a
scale that runs on a single GPU in a few hours.

Usage:
    python scripts/train_gpt_demo.py
    python scripts/train_gpt_demo.py --n_layer 6 --n_embd 384 --steps 2000
    python scripts/train_gpt_demo.py --dataset wikitext  # needs datasets lib
"""

import argparse
import math
import os
import time
import json
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attnres.models.gpt_demo import GPTConfig, GPTWithAttnRes
from attnres.visualisation.scaling_laws import fit_and_plot_scaling_law
from attnres.visualisation.training_dynamics import plot_training_dynamics


# ---------------------------------------------------------------------------
# Synthetic data fallback — character-level sequences on random text
# ---------------------------------------------------------------------------

def make_synthetic_dataset(vocab_size: int, seq_len: int, n_samples: int):
    """Random token sequences for smoke-test / unit demonstration."""
    torch.manual_seed(42)
    data = torch.randint(0, vocab_size, (n_samples, seq_len + 1))
    x = data[:, :-1]
    y = data[:, 1:]
    return TensorDataset(x, y)


def make_wikitext_dataset(seq_len: int, split: str = "train", max_tokens: int = 5_000_000):
    """Character-level WikiText-103 encoded as token IDs (0-255)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
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
    cfg: GPTConfig,
    dataset,
    steps: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    log_interval: int = 50,
) -> Dict:
    model = GPTWithAttnRes(cfg).to(device)
    n_params = model.num_params()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                                   betas=(0.9, 0.95))

    # Cosine LR with warmup
    warmup = min(100, steps // 10)
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(steps - warmup, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log = {"steps": [], "losses": [], "output_magnitudes": {}, "grad_norms": {}}
    model.train()
    step = 0
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
            print(f"  step {step:5d}/{steps} | loss {loss.item():.4f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")
            log["steps"].append(step)
            log["losses"].append(loss.item())
            t0 = time.perf_counter()

    # Capture final per-layer gradient norms
    for i, block in enumerate(model.transformer.h):
        grads = [p.grad for p in block.parameters() if p.grad is not None]
        if grads:
            norm = sum(g.float().norm() ** 2 for g in grads) ** 0.5
            log["grad_norms"][i] = norm.item()

    log["n_params"] = n_params
    return log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layer",   type=int,   default=6)
    parser.add_argument("--n_embd",    type=int,   default=384)
    parser.add_argument("--n_head",    type=int,   default=6)
    parser.add_argument("--block_size",type=int,   default=256)
    parser.add_argument("--steps",     type=int,   default=3000)
    parser.add_argument("--batch_size",type=int,   default=32)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--num_attnres_blocks", type=int, default=4)
    parser.add_argument("--dataset",   type=str,   default="synthetic",
                        choices=["synthetic", "wikitext"])
    parser.add_argument("--out_dir",   type=str,   default="./outputs")
    parser.add_argument("--device",    type=str,   default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    vocab_size = 256  # byte-level

    print(f"Device: {device}")
    print(f"Building dataset ({args.dataset}) ...")
    if args.dataset == "wikitext":
        dataset = make_wikitext_dataset(args.block_size)
    else:
        dataset = make_synthetic_dataset(vocab_size, args.block_size, n_samples=8000)

    base_cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        use_attnres=False,
    )

    attnres_cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        use_attnres=True,
        num_attnres_blocks=args.num_attnres_blocks,
    )

    print(f"\n--- Training BASELINE (use_attnres=False) ---")
    baseline_log = train(base_cfg, dataset, args.steps, args.lr,
                         args.batch_size, device)

    print(f"\n--- Training ATTNRES (use_attnres=True, N={args.num_attnres_blocks}) ---")
    attnres_log = train(attnres_cfg, dataset, args.steps, args.lr,
                        args.batch_size, device)

    # Save logs
    with open(os.path.join(args.out_dir, "baseline_log.json"), "w") as f:
        json.dump(baseline_log, f, indent=2)
    with open(os.path.join(args.out_dir, "attnres_log.json"), "w") as f:
        json.dump(attnres_log, f, indent=2)

    # Plot training dynamics
    plot_training_dynamics(
        log_dict=attnres_log,
        baseline_log_dict=baseline_log,
        save_path=os.path.join(args.out_dir, "training_dynamics.png"),
    )

    # Plot scaling law (single-point, illustrative)
    n_params = attnres_log["n_params"]
    flops_per_step = 6 * n_params * args.block_size * args.batch_size
    total_flops_b = [flops_per_step * args.steps / 1e15]
    fit_and_plot_scaling_law(
        compute_flops=[total_flops_b, total_flops_b],
        val_losses=[[baseline_log["losses"][-1]], [attnres_log["losses"][-1]]],
        labels=["Baseline", "Block AttnRes"],
        save_path=os.path.join(args.out_dir, "scaling_laws.png"),
    )

    final_baseline = baseline_log["losses"][-1]
    final_attnres  = attnres_log["losses"][-1]
    print(f"\n{'='*50}")
    print(f"Final loss — Baseline:    {final_baseline:.4f}")
    print(f"Final loss — Block AttnRes: {final_attnres:.4f}")
    print(f"Delta: {final_baseline - final_attnres:+.4f} "
          f"({'AttnRes better' if final_attnres < final_baseline else 'Baseline better'})")
    print(f"Outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

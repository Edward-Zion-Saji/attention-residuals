"""
Multi-scale evaluation: reproduce Fig 4 (scaling laws) from arXiv:2603.15031
at a scale that runs on a single CPU/GPU in under 30 minutes.

Four model sizes (matching the spirit of Table 2 in the paper):
  Tiny   ~0.4M params  — 2 layers, d=64
  Small  ~1.5M params  — 4 layers, d=128
  Medium ~6M  params   — 6 layers, d=256
  Large  ~12M params   — 8 layers, d=384

Each size is trained three ways:
  1. Baseline (standard residuals)
  2. Block AttnRes (N=4 blocks)
  3. Full AttnRes

Results go into outputs/eval_scaling/ as JSON logs and PNG plots.

Usage:
    python scripts/eval_scaling.py
    python scripts/eval_scaling.py --quick   # 1000 steps per model
    python scripts/eval_scaling.py --dataset wikitext
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attnres.models.gpt_demo import GPTConfig, GPTWithAttnRes
from attnres.visualisation.scaling_laws import fit_and_plot_scaling_law
from attnres.visualisation.training_dynamics import plot_training_dynamics


# ---------------------------------------------------------------------------
# Dataset helpers (same as train_gpt_demo.py)
# ---------------------------------------------------------------------------

def make_synthetic_dataset(vocab_size: int, seq_len: int, n_samples: int):
    torch.manual_seed(42)
    data = torch.randint(0, vocab_size, (n_samples, seq_len + 1))
    x = data[:, :-1]
    y = data[:, 1:]
    return TensorDataset(x, y)


def make_wikitext_dataset(seq_len: int, split: str = "train", max_tokens: int = 3_000_000):
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
# Training loop — returns a structured log dict
# ---------------------------------------------------------------------------

def train(
    cfg: GPTConfig,
    dataset,
    steps: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    log_interval: int = 100,
    warmup_frac: float = 0.05,
) -> Dict:
    model = GPTWithAttnRes(cfg).to(device)
    n_params = model.num_params()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95)
    )

    warmup = max(50, int(steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(steps - warmup, 1)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log = {
        "steps": [],
        "losses": [],
        "output_magnitudes": {},
        "grad_norms": {},
        "n_params": n_params,
        "variant": "attnres" if cfg.use_attnres else "baseline",
        "n_layer": cfg.n_layer,
        "n_embd": cfg.n_embd,
    }

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
            print(
                f"  step {step:5d}/{steps} | loss {loss.item():.4f} | "
                f"lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
            )
            log["steps"].append(step)
            log["losses"].append(loss.item())
            t0 = time.perf_counter()

    # Capture final per-block output magnitudes and gradient norms
    # We run one forward pass on a fixed batch to get output magnitudes
    model.eval()
    with torch.no_grad():
        try:
            x_fixed, _ = next(iter(loader))
        except StopIteration:
            x_fixed = torch.randint(0, cfg.vocab_size, (4, cfg.block_size), device=device)
        x_fixed = x_fixed[:4].to(device)

        # Hook to capture output magnitudes
        block_magnitudes = {}
        hooks = []
        for i, block in enumerate(model.transformer.h):
            def make_hook(idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0]
                    else:
                        out = output
                    block_magnitudes[idx] = out.float().norm(dim=-1).mean().item()
                return hook
            hooks.append(block.register_forward_hook(make_hook(i)))

        _ = model(x_fixed)
        for h in hooks:
            h.remove()

    log["output_magnitudes"] = {str(k): v for k, v in block_magnitudes.items()}

    # Gradient norms — do one backward pass
    model.train()
    x_fixed, y_fixed = next(iter(loader))
    x_fixed, y_fixed = x_fixed[:4].to(device), y_fixed[:4].to(device)
    _, loss_fixed = model(x_fixed, y_fixed)
    loss_fixed.backward()

    for i, block in enumerate(model.transformer.h):
        grads = [p.grad for p in block.parameters() if p.grad is not None]
        if grads:
            norm = sum(g.float().norm() ** 2 for g in grads) ** 0.5
            log["grad_norms"][str(i)] = norm.item()

    optimizer.zero_grad()
    return log


# ---------------------------------------------------------------------------
# Model size configurations
# ---------------------------------------------------------------------------

MODEL_SIZES = [
    # (name,  n_layer, n_embd, n_head, steps, batch_size, lr)
    ("tiny",   4,  128,  4,  2000, 32, 3e-4),
    ("small",  6,  256,  4,  3000, 32, 3e-4),
    ("medium", 8,  384,  6,  4000, 32, 2e-4),
    ("large",  12, 512,  8,  5000, 32, 1.5e-4),
]

MODEL_SIZES_QUICK = [
    ("tiny",   4,  128,  4,  800,  16, 3e-4),
    ("small",  6,  256,  4,  1000, 16, 3e-4),
    ("medium", 8,  384,  6,  1200, 16, 2e-4),
    ("large",  12, 512,  8,  1500, 16, 1.5e-4),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Fewer steps — for a fast sanity-check run")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "wikitext"])
    parser.add_argument("--block_size", type=int, default=128,
                        help="Sequence length / context window")
    parser.add_argument("--num_attnres_blocks", type=int, default=4,
                        help="N for Block AttnRes")
    parser.add_argument("--out_dir", type=str, default="./outputs/eval_scaling")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip_full", action="store_true",
                        help="Skip Full AttnRes (only Baseline + Block AttnRes)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    vocab_size = 256

    sizes = MODEL_SIZES_QUICK if args.quick else MODEL_SIZES

    print(f"\n{'='*60}")
    print(f"AttnRes Scaling Evaluation")
    print(f"Device:  {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Sizes:   {[s[0] for s in sizes]}")
    print(f"{'='*60}\n")

    if args.dataset == "wikitext":
        dataset = make_wikitext_dataset(args.block_size)
    else:
        dataset = make_synthetic_dataset(vocab_size, args.block_size, n_samples=6000)

    # Collect results across sizes
    all_compute = {"baseline": [], "block_attnres": [], "full_attnres": []}
    all_losses  = {"baseline": [], "block_attnres": [], "full_attnres": []}
    all_params  = {"baseline": [], "block_attnres": [], "full_attnres": []}
    size_results = []

    for name, n_layer, n_embd, n_head, steps, batch_size, lr in sizes:
        print(f"\n{'─'*60}")
        print(f"Model size: {name}  (L={n_layer}, d={n_embd}, steps={steps})")
        print(f"{'─'*60}")

        row = {"size": name, "n_layer": n_layer, "n_embd": n_embd}

        for variant, use_attnres, attnres_variant in [
            ("Baseline",       False, "block"),
            ("Block AttnRes",  True,  "block"),
            ("Full AttnRes",   True,  "full"),
        ]:
            if attnres_variant == "full" and args.skip_full:
                continue

            print(f"\n  --- {variant} ---")
            cfg = GPTConfig(
                vocab_size=vocab_size,
                block_size=args.block_size,
                n_layer=n_layer,
                n_embd=n_embd,
                n_head=n_head,
                use_attnres=use_attnres,
                num_attnres_blocks=args.num_attnres_blocks,
                attnres_variant=attnres_variant,
            )
            log = train(cfg, dataset, steps, lr, batch_size, device)

            # Save individual log
            slug = variant.lower().replace(" ", "_")
            log_path = os.path.join(args.out_dir, f"{name}_{slug}.json")
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)

            final_loss = log["losses"][-1] if log["losses"] else float("nan")
            n_params = log["n_params"]

            # Compute FLOPs per step: 6 * params * seq_len
            flops_per_step = 6 * n_params * args.block_size
            total_flops_pflops = flops_per_step * steps * batch_size / 1e15

            key_map = {
                "Baseline": "baseline",
                "Block AttnRes": "block_attnres",
                "Full AttnRes": "full_attnres",
            }
            key = key_map[variant]
            all_compute[key].append(total_flops_pflops)
            all_losses[key].append(final_loss)
            all_params[key].append(n_params)

            row[f"{slug}_loss"] = round(final_loss, 4)
            row[f"{slug}_params"] = n_params

            print(f"  Final loss: {final_loss:.4f}  |  params: {n_params:,}  |  FLOPs: {total_flops_pflops:.3e} PF")

            # Training dynamics plot for this size + variant combo
            if variant == "Baseline":
                baseline_log_for_size = log
            if variant == "Block AttnRes":
                dyn_path = os.path.join(args.out_dir, f"{name}_training_dynamics.png")
                plot_training_dynamics(
                    log_dict=log,
                    baseline_log_dict=baseline_log_for_size if "baseline_log_for_size" in dir() else None,
                    save_path=dyn_path,
                    label=f"Block AttnRes ({name})",
                    baseline_label=f"Baseline ({name})",
                )

        size_results.append(row)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SCALING EVALUATION SUMMARY")
    print(f"{'='*70}")
    header = f"{'Size':8s} {'Params':>10s} {'Baseline':>10s} {'Block AR':>10s} {'Full AR':>10s} {'ΔBlock':>8s} {'ΔFull':>8s}"
    print(header)
    print("─" * len(header))

    for row in size_results:
        bl  = row.get("baseline_loss", float("nan"))
        bar = row.get("block_attnres_loss", float("nan"))
        far = row.get("full_attnres_loss", float("nan"))
        params = row.get("baseline_params", 0)
        delta_block = bl - bar
        delta_full  = bl - far if not math.isnan(far) else float("nan")
        print(
            f"{row['size']:8s} {params:>10,} {bl:>10.4f} {bar:>10.4f} "
            f"{far:>10.4f} {delta_block:>+8.4f} {delta_full:>+8.4f}"
        )

    # ------------------------------------------------------------------
    # Scaling law plots
    # ------------------------------------------------------------------
    variants_to_plot = ["baseline", "block_attnres"]
    labels_to_plot = ["Baseline", "Block AttnRes"]
    colors_to_plot = ["#d6604d", "#2166ac"]

    if not args.skip_full and all_compute["full_attnres"]:
        variants_to_plot.append("full_attnres")
        labels_to_plot.append("Full AttnRes")
        colors_to_plot.append("#4dac26")

    compute_lists = [all_compute[v] for v in variants_to_plot]
    loss_lists    = [all_losses[v]  for v in variants_to_plot]

    scaling_path = os.path.join(args.out_dir, "scaling_laws.png")
    fit_and_plot_scaling_law(
        compute_flops=compute_lists,
        val_losses=loss_lists,
        labels=labels_to_plot,
        colors=colors_to_plot,
        save_path=scaling_path,
    )

    # ------------------------------------------------------------------
    # Save machine-readable summary
    # ------------------------------------------------------------------
    summary = {
        "device": str(device),
        "dataset": args.dataset,
        "block_size": args.block_size,
        "num_attnres_blocks": args.num_attnres_blocks,
        "size_results": size_results,
        "compute": {v: all_compute[v] for v in variants_to_plot},
        "losses":  {v: all_losses[v]  for v in variants_to_plot},
    }
    summary_path = os.path.join(args.out_dir, "scaling_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs saved to {args.out_dir}/")
    print(f"  scaling_laws.png")
    print(f"  scaling_summary.json")
    print(f"  <size>_training_dynamics.png (per model size)")
    print(f"  <size>_<variant>.json        (per model logs)")


if __name__ == "__main__":
    main()

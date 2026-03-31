"""
Self-contained experiment runner that trains 4 model sizes × 3 variants
and writes all results to a JSON file.  Designed to run start-to-finish
on a CPU in 15-20 minutes using very small models.

Run:
    python3 scripts/run_experiments.py
    python3 scripts/run_experiments.py --mode ablation
    python3 scripts/run_experiments.py --mode depth
"""
import argparse, json, math, os, sys, time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attnres.models.gpt_demo import GPTConfig, GPTWithAttnRes
from attnres.core.block_attn_res import BlockAttnRes
from attnres.core.utils import RMSNorm, zero_init_
from attnres.visualisation.scaling_laws import fit_and_plot_scaling_law
from attnres.visualisation.training_dynamics import plot_training_dynamics
from attnres.visualisation.attention_maps import compute_attention_weights, plot_depth_attention

# --------------------------------------------------------------------------- #
# Reproducible synthetic dataset
# --------------------------------------------------------------------------- #
def make_dataset(vocab_size=256, seq_len=128, n_samples=4000):
    torch.manual_seed(42)
    data = torch.randint(0, vocab_size, (n_samples, seq_len + 1))
    return TensorDataset(data[:, :-1], data[:, 1:])

# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(model, dataset, steps, lr=3e-4, batch=32, device="cpu",
          report_every=100, label=""):
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                             betas=(0.9, 0.95))
    warmup = max(30, steps // 15)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s/warmup if s<warmup else
        max(0.05, 0.5*(1+math.cos(math.pi*(s-warmup)/max(steps-warmup,1))))
    )
    model.train()
    it = iter(loader)
    log_steps, log_losses = [], []
    t0 = time.perf_counter()
    for step in range(1, steps+1):
        try: x,y = next(it)
        except StopIteration:
            it = iter(loader); x,y = next(it)
        x,y = x.to(device), y.to(device)
        _,loss = model(x,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if step % report_every == 0:
            elapsed = time.perf_counter()-t0
            print(f"  [{label}] step {step:4d}/{steps}  loss={loss.item():.4f}  {elapsed:.1f}s")
            log_steps.append(step); log_losses.append(loss.item())
            t0 = time.perf_counter()

    # per-block output magnitudes
    magnitudes = {}
    model.eval()
    with torch.no_grad():
        xf = next(iter(loader))[0][:8].to(device)
        hooks = []
        for i, blk in enumerate(model.transformer.h):
            def make_hook(idx):
                def h(m,inp,out): magnitudes[str(idx)] = out.float().norm(dim=-1).mean().item()
                return h
            hooks.append(blk.register_forward_hook(make_hook(i)))
        model(xf)
        for h in hooks: h.remove()

    # per-block gradient norms
    grad_norms = {}
    model.train()
    xg,yg = next(iter(loader))
    xg,yg = xg[:8].to(device), yg[:8].to(device)
    _,lg = model(xg,yg); lg.backward()
    for i, blk in enumerate(model.transformer.h):
        gs = [p.grad for p in blk.parameters() if p.grad is not None]
        if gs:
            n = sum(g.float().norm()**2 for g in gs)**0.5
            grad_norms[str(i)] = n.item()
    opt.zero_grad()

    return {
        "steps": log_steps, "losses": log_losses,
        "output_magnitudes": magnitudes, "grad_norms": grad_norms,
        "final_loss": log_losses[-1] if log_losses else float("nan"),
        "n_params": sum(p.numel() for p in model.parameters()),
    }, model

# --------------------------------------------------------------------------- #
# Ablation: sigmoid variant
# --------------------------------------------------------------------------- #
class BlockAttnResSigmoid(BlockAttnRes):
    def compute_all_inputs(self, layer_outputs, embedding):
        L = len(layer_outputs)
        inputs, block_reprs, partial = [], [embedding], None
        for l in range(L):
            bn = self._block_of(l); i = l - self._block_starts[bn]
            sources = list(block_reprs)
            if i > 0 and partial is not None: sources.append(partial)
            v = torch.stack(sources, dim=2)
            k = self.key_norm(v); w = self.queries[l]
            sc = (k*w).sum(-1)
            alpha = torch.sigmoid(sc)
            alpha = alpha / (alpha.sum(-1, keepdim=True)+1e-8)
            inputs.append((alpha.unsqueeze(-1)*v).sum(2))
            fl = layer_outputs[l]
            partial = fl if partial is None else partial+fl
            if i+1 >= self.block_sizes[bn]:
                block_reprs.append(partial); partial = None
        return inputs

class BlockAttnResNoNorm(BlockAttnRes):
    def compute_all_inputs(self, layer_outputs, embedding):
        L = len(layer_outputs)
        inputs, block_reprs, partial = [], [embedding], None
        for l in range(L):
            bn = self._block_of(l); i = l - self._block_starts[bn]
            sources = list(block_reprs)
            if i > 0 and partial is not None: sources.append(partial)
            v = torch.stack(sources, dim=2)
            w = self.queries[l]
            sc = (v*w).sum(-1)   # no RMSNorm
            alpha = torch.softmax(sc, dim=-1)
            inputs.append((alpha.unsqueeze(-1)*v).sum(2))
            fl = layer_outputs[l]
            partial = fl if partial is None else partial+fl
            if i+1 >= self.block_sizes[bn]:
                block_reprs.append(partial); partial = None
        return inputs

# --------------------------------------------------------------------------- #
# SCALING LAW EXPERIMENT
# --------------------------------------------------------------------------- #
def run_scaling(out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    dataset = make_dataset()
    vocab_size = 256

    # (name, n_layer, n_embd, n_head, steps, lr, batch)
    # Sized to finish in ~30 min on Apple M-series with MPS.
    # The key insight is block_attnres is ~2x slower than baseline
    # due to the Python-level double pass in compute_all_inputs.
    # tiny:  baseline ~0.09s/step, block ~0.18s/step  → 3000 steps ≈ 9 min
    # small: baseline ~0.25s/step, block ~0.50s/step  → 2000 steps ≈ 17 min
    # medium: block ~1.1s/step → 2000 steps ≈ 37 min — too slow, skip
    sizes = [
        ("tiny",  4, 128, 4, 3000, 3e-4, 64),
        ("small", 6, 256, 4, 2000, 3e-4, 64),
    ]

    all_results = {}
    collect_c = {"baseline": [], "block_attnres": []}
    collect_l = {"baseline": [], "block_attnres": []}

    for name, n_layer, n_embd, n_head, steps, lr, batch in sizes:
        print(f"\n{'─'*55}")
        print(f"Model: {name}  (L={n_layer}, d={n_embd}, steps={steps})")
        print(f"{'─'*55}")
        row = {}

        for variant, use_ar, ar_var in [
            ("baseline",       False, "block"),
            ("block_attnres",  True,  "block"),
        ]:
            label = f"{name}/{variant}"
            cfg = GPTConfig(
                vocab_size=vocab_size, block_size=128,
                n_layer=n_layer, n_embd=n_embd, n_head=n_head,
                use_attnres=use_ar, num_attnres_blocks=4,
                attnres_variant=ar_var,
            )
            log, model = train(GPTWithAttnRes(cfg), dataset, steps, lr, batch,
                               device, report_every=max(steps//8, 50), label=label)
            row[variant] = log
            # flops: 6 * params * seq_len * batch * steps
            flops = 6 * log["n_params"] * 128 * batch * steps / 1e15
            collect_c[variant].append(flops)
            collect_l[variant].append(log["final_loss"])
            print(f"  → final loss: {log['final_loss']:.4f}  params: {log['n_params']:,}")

        # training dynamics plot
        dyn_path = os.path.join(out_dir, f"{name}_dynamics.png")
        plot_training_dynamics(
            row["block_attnres"], baseline_log_dict=row["baseline"],
            save_path=dyn_path, label=f"Block AttnRes ({name})",
            baseline_label=f"Baseline ({name})"
        )

        all_results[name] = {
            "n_layer": n_layer, "n_embd": n_embd,
            "baseline_loss": row["baseline"]["final_loss"],
            "block_attnres_loss": row["block_attnres"]["final_loss"],
            "baseline_params": row["baseline"]["n_params"],
        }

    # scaling law plot
    sc_path = os.path.join(out_dir, "scaling_laws.png")
    fit_and_plot_scaling_law(
        compute_flops=[collect_c["baseline"], collect_c["block_attnres"]],
        val_losses=[collect_l["baseline"], collect_l["block_attnres"]],
        labels=["Baseline", "Block AttnRes (N=4)"],
        colors=["#d6604d", "#2166ac"],
        save_path=sc_path,
    )

    with open(os.path.join(out_dir, "scaling_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*55}")
    print("SCALING RESULTS")
    print(f"{'='*55}")
    print(f"{'Size':8s} {'Params':>10s} {'Baseline':>10s} {'Block AR':>10s} {'Δ':>8s}")
    print("─"*50)
    for name, r in all_results.items():
        d = r["baseline_loss"] - r["block_attnres_loss"]
        print(f"{name:8s} {r['baseline_params']:>10,} "
              f"{r['baseline_loss']:>10.4f} {r['block_attnres_loss']:>10.4f} {d:>+8.4f}")

    return all_results

# --------------------------------------------------------------------------- #
# ABLATION EXPERIMENT
# --------------------------------------------------------------------------- #
def run_ablation(out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    dataset = make_dataset()
    vocab_size = 256
    n_layer, n_embd, n_head = 6, 256, 4   # small model: ~0.5s/step on MPS
    steps, lr, batch = 2000, 3e-4, 64    # 7 variants × 2000 steps × 0.5s ≈ 23 min
    num_ar_layers = n_layer * 2

    def make_cfg(**kw):
        return GPTConfig(vocab_size=vocab_size, block_size=128,
                         n_layer=n_layer, n_embd=n_embd, n_head=n_head, **kw)

    variants = []
    # 1. Baseline
    variants.append(("Baseline", GPTWithAttnRes(make_cfg(use_attnres=False))))

    # 2-4. Block AttnRes N=2,4,8
    for N in [2, 4, 8]:
        m = GPTWithAttnRes(make_cfg(use_attnres=True, num_attnres_blocks=N))
        variants.append((f"Block AttnRes N={N}", m))

    # 5. Full AttnRes
    variants.append(("Full AttnRes",
        GPTWithAttnRes(make_cfg(use_attnres=True, num_attnres_blocks=num_ar_layers,
                                attnres_variant="full"))))

    # 6. sigmoid
    m_sig = GPTWithAttnRes(make_cfg(use_attnres=True, num_attnres_blocks=4))
    m_sig.attnres = BlockAttnResSigmoid(num_ar_layers, n_embd, num_blocks=4)
    variants.append(("Block AttnRes (sigmoid)", m_sig))

    # 7. no RMSNorm
    m_nn = GPTWithAttnRes(make_cfg(use_attnres=True, num_attnres_blocks=4))
    m_nn.attnres = BlockAttnResNoNorm(num_ar_layers, n_embd, num_blocks=4)
    variants.append(("Block AttnRes (no RMSNorm)", m_nn))

    results = []
    all_logs = []
    for name, model in variants:
        print(f"\n--- {name} ---")
        log, _ = train(model, dataset, steps, lr, batch, device,
                       report_every=max(steps//6, 50), label=name)
        results.append({"name": name, "final_loss": log["final_loss"],
                         "n_params": log["n_params"]})
        all_logs.append((name, log))
        print(f"  final loss: {log['final_loss']:.4f}")

    baseline_loss = next(r["final_loss"] for r in results if r["name"] == "Baseline")

    print(f"\n{'='*65}")
    print("ABLATION RESULTS  (Table 4 style)")
    print(f"{'='*65}")
    print(f"{'Variant':<38} {'Loss':>8}  {'Δ vs Baseline':>14}")
    print("─"*65)
    for r in results:
        d = baseline_loss - r["final_loss"]
        sign = f"+{d:.4f}" if d > 0 else f"{d:.4f}"
        print(f"{r['name']:<38} {r['final_loss']:>8.4f}  {sign:>14}")

    # loss curve plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#d6604d","#2166ac","#1a9850","#7b3294","#e08214","#006837","#a6d96a"]
    fig, ax = plt.subplots(figsize=(10,5))
    for (name, log), color in zip(all_logs, colors):
        if log["steps"] and log["losses"]:
            ax.plot(log["steps"], log["losses"], lw=1.5, label=name, color=color)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Ablation: Loss Curves for All Variants")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_curves.png"), dpi=150)
    plt.close(fig)

    # bar chart
    names = [r["name"] for r in results]
    losses = [r["final_loss"] for r in results]
    fig, ax = plt.subplots(figsize=(max(10, len(names)*1.4), 5))
    bar_colors = ["#d6604d" if "Baseline"==n else "#2166ac" for n in names]
    bars = ax.bar(range(len(names)), losses, color=bar_colors, width=0.6)
    for bar, l in zip(bars, losses):
        ax.text(bar.get_x()+bar.get_width()/2, l+(max(losses)-min(losses))*0.01,
                f"{l:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Final Loss"); ax.set_title("Ablation Study — Key AttnRes Design Choices")
    ax.axhline(baseline_loss, color="#d6604d", linestyle="--", lw=1.2, alpha=0.7,
               label=f"Baseline ({baseline_loss:.4f})")
    ax.legend(fontsize=9)
    lr_range = max(losses)-min(losses)
    ax.set_ylim(min(losses)-lr_range*0.3, max(losses)+lr_range*0.3)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_bars.png"), dpi=150)
    plt.close(fig)
    print(f"Saved plots to {out_dir}/")

    with open(os.path.join(out_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results

# --------------------------------------------------------------------------- #
# DEPTH ATTENTION EXPERIMENT
# --------------------------------------------------------------------------- #
def run_depth(out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    dataset = make_dataset()
    vocab_size = 256
    n_layer, n_embd, n_head = 6, 256, 4   # small model
    steps, lr, batch = 2000, 3e-4, 64    # 2 variants × 2000 steps × 0.5s ≈ 33 min

    depth_stats = []

    for variant, use_ar, ar_var, N in [
        ("Block AttnRes", True, "block", 4),
        ("Full AttnRes",  True, "full",  n_layer*2),
    ]:
        cfg = GPTConfig(vocab_size=vocab_size, block_size=128, n_layer=n_layer,
                         n_embd=n_embd, n_head=n_head, use_attnres=use_ar,
                         num_attnres_blocks=N, attnres_variant=ar_var)
        print(f"\n--- Training {variant} for depth analysis ---")
        log, model = train(GPTWithAttnRes(cfg), dataset, steps, lr, batch, device,
                            report_every=250, label=variant)
        print(f"  final loss: {log['final_loss']:.4f}")

        slug = variant.lower().replace(" ", "_")

        # Standard depth attention heatmap
        hm_path = os.path.join(out_dir, f"{slug}_heatmap.png")
        plot_depth_attention(model, save_path=hm_path,
                              title=f"α_{{i→l}} Heatmap — {variant} (trained)")

        # Compute & analyse weights
        weights = compute_attention_weights(model.attnres)
        import numpy as np
        W = weights
        # Diagonal dominance: last valid column per row
        diag_vals = []
        for l in range(W.shape[0]):
            row = W[l]
            valid_idx = np.where(~np.isnan(row))[0]
            if len(valid_idx):
                diag_vals.append(row[valid_idx[-1]])

        # Embedding persistence: column 0 weight per row
        emb_vals = W[:, 0]; emb_vals = emb_vals[~np.isnan(emb_vals)]

        # Attention entropy
        entropies = []
        for l in range(W.shape[0]):
            row = W[l]; valid = row[~np.isnan(row)]
            if len(valid)>1:
                entropies.append(-np.sum(valid*np.log(valid+1e-9)))

        st = {
            "variant": variant,
            "final_loss": log["final_loss"],
            "mean_diagonal_weight": float(np.mean(diag_vals)) if diag_vals else float("nan"),
            "mean_embedding_weight": float(np.mean(emb_vals)) if len(emb_vals) else float("nan"),
            "mean_entropy":  float(np.mean(entropies))  if entropies else float("nan"),
        }
        depth_stats.append(st)
        print(f"  Diagonal weight (locality):    {st['mean_diagonal_weight']:.4f}")
        print(f"  Embedding weight (persistence): {st['mean_embedding_weight']:.4f}")
        print(f"  Mean attention entropy:         {st['mean_entropy']:.4f}")

    with open(os.path.join(out_dir, "depth_stats.json"), "w") as f:
        json.dump(depth_stats, f, indent=2)
    return depth_stats

# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scaling","ablation","depth","all"],
                        default="all")
    parser.add_argument("--out_dir", default="./outputs/experiments")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f" AttnRes Evaluation  |  device={device}  |  mode={args.mode}")
    print(f"{'='*55}")

    all_data = {}

    if args.mode in ("scaling", "all"):
        print("\n\n[1/3] SCALING LAW EXPERIMENT")
        r = run_scaling(os.path.join(args.out_dir, "scaling"), device)
        all_data["scaling"] = r

    if args.mode in ("ablation", "all"):
        print("\n\n[2/3] ABLATION EXPERIMENT")
        r = run_ablation(os.path.join(args.out_dir, "ablation"), device)
        all_data["ablation"] = r

    if args.mode in ("depth", "all"):
        print("\n\n[3/3] DEPTH ATTENTION EXPERIMENT")
        r = run_depth(os.path.join(args.out_dir, "depth"), device)
        all_data["depth"] = r

    with open(os.path.join(args.out_dir, "all_results.json"), "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\n\nAll done. Results written to {args.out_dir}/all_results.json")

if __name__ == "__main__":
    main()

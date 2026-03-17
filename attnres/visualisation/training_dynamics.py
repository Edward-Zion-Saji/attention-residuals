"""
Training dynamics visualisation (reproduces Fig 5 from arXiv:2603.15031).

Three-panel plot:
  (a) Validation loss over training steps
  (b) Per-layer output magnitude at end of training
  (c) Per-layer gradient norm magnitude

Shows that AttnRes mitigates PreNorm dilution:
  - Output magnitudes stay bounded (periodic within blocks) vs monotone growth
  - Gradient norms are more uniform vs disproportionately large at early layers
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_dynamics(
    log_dict: Dict[str, Any],
    save_path: str = "training_dynamics.png",
    baseline_log_dict: Optional[Dict[str, Any]] = None,
    figsize: tuple = (15, 5),
    label: str = "Block AttnRes",
    baseline_label: str = "Baseline",
) -> None:
    """Plot training dynamics: loss curve, output magnitudes, gradient norms.

    Reproduces Fig 5 of arXiv:2603.15031.

    Args:
        log_dict:          Training log from Qwen35AttnResTrainer or GPT demo.
                           Expected keys: "steps", "losses", "output_magnitudes",
                           "grad_norms". output_magnitudes and grad_norms are
                           dicts mapping layer_idx -> value (final snapshot).
        save_path:         Output file path.
        baseline_log_dict: Optional matching log for a baseline (no AttnRes) run.
        figsize:           Matplotlib figure size.
        label:             Legend label for the AttnRes model.
        baseline_label:    Legend label for the baseline model.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    _COLOR_ATTNRES  = "#2166ac"
    _COLOR_BASELINE = "#d6604d"

    # ------------------------------------------------------------------
    # Panel (a): Validation loss
    # ------------------------------------------------------------------
    ax = axes[0]
    steps = log_dict.get("steps", [])
    losses = log_dict.get("losses", [])
    if steps and losses:
        ax.plot(steps, losses, color=_COLOR_ATTNRES, linewidth=1.5, label=label)
    if baseline_log_dict:
        b_steps = baseline_log_dict.get("steps", [])
        b_losses = baseline_log_dict.get("losses", [])
        if b_steps and b_losses:
            ax.plot(b_steps, b_losses, color=_COLOR_BASELINE, linewidth=1.5,
                    linestyle="--", label=baseline_label)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss")
    ax.set_title("(a) Validation Loss")
    handles, lbls = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel (b): Per-layer output magnitude
    # ------------------------------------------------------------------
    ax = axes[1]
    magnitudes = log_dict.get("output_magnitudes", {})
    if magnitudes:
        layer_ids = sorted(int(k) for k in magnitudes.keys())
        mags = [magnitudes[str(k)] if str(k) in magnitudes else magnitudes[k]
                for k in layer_ids]
        ax.bar(layer_ids, mags, color=_COLOR_ATTNRES, alpha=0.8,
               width=0.8, label=label)
    if baseline_log_dict:
        b_mags = baseline_log_dict.get("output_magnitudes", {})
        if b_mags:
            b_layer_ids = sorted(int(k) for k in b_mags.keys())
            b_vals = [b_mags[str(k)] if str(k) in b_mags else b_mags[k]
                      for k in b_layer_ids]
            ax.plot(b_layer_ids, b_vals, color=_COLOR_BASELINE, linewidth=1.5,
                    linestyle="--", marker="o", markersize=3, label=baseline_label)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Output magnitude (L2 norm)")
    ax.set_title("(b) Output Magnitude per Layer")
    handles, lbls = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ------------------------------------------------------------------
    # Panel (c): Per-layer gradient norm
    # ------------------------------------------------------------------
    ax = axes[2]
    grad_norms = log_dict.get("grad_norms", {})
    if grad_norms:
        layer_ids = sorted(int(k) for k in grad_norms.keys())
        gnorms = [grad_norms[str(k)] if str(k) in grad_norms else grad_norms[k]
                  for k in layer_ids]
        ax.bar(layer_ids, gnorms, color=_COLOR_ATTNRES, alpha=0.8,
               width=0.8, label=label)
    if baseline_log_dict:
        b_gnorms = baseline_log_dict.get("grad_norms", {})
        if b_gnorms:
            b_layer_ids = sorted(int(k) for k in b_gnorms.keys())
            b_vals = [b_gnorms[str(k)] if str(k) in b_gnorms else b_gnorms[k]
                      for k in b_layer_ids]
            ax.plot(b_layer_ids, b_vals, color=_COLOR_BASELINE, linewidth=1.5,
                    linestyle="--", marker="o", markersize=3, label=baseline_label)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Gradient norm")
    ax.set_title("(c) Gradient Norm per Layer")
    handles, lbls = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Training Dynamics: AttnRes vs Baseline", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training dynamics plot → {save_path}")


def load_log_from_json(path: str) -> Dict[str, Any]:
    """Load a training log JSON saved by Qwen35AttnResTrainer or train_gpt_demo.py.

    Converts the flat list-of-dicts format into the panel-ready dict format.

    Args:
        path: Path to log_history.json.

    Returns:
        Dict with keys: steps, losses, output_magnitudes, grad_norms.
    """
    with open(path) as f:
        raw = json.load(f)

    steps, losses = [], []
    output_magnitudes: Dict[int, float] = {}
    grad_norms: Dict[int, float] = {}

    for entry in raw:
        steps.append(entry.get("step", 0))
        losses.append(entry.get("loss", float("nan")))
        # Keep last snapshot of magnitudes / grad norms
        for k, v in entry.get("output_magnitudes", {}).items():
            output_magnitudes[int(k)] = v
        for k, v in entry.get("grad_norms", {}).items():
            grad_norms[int(k)] = v

    return {
        "steps": steps,
        "losses": losses,
        "output_magnitudes": output_magnitudes,
        "grad_norms": grad_norms,
    }

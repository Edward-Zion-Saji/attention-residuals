"""
Scaling law visualisation (reproduces Fig 4 from arXiv:2603.15031).

Fits power-law curves L = A * C^{-alpha} to (compute, loss) pairs for
baseline and AttnRes variants, then plots them on a log-log scale.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Optional


def fit_power_law(compute: List[float], losses: List[float]):
    """Fit L = A * C^{-alpha} via linear regression in log-log space.

    Args:
        compute: List of compute values (e.g. PFLOP/s-days).
        losses:  Corresponding validation losses.

    Returns:
        (A, alpha): Power-law coefficients.
    """
    log_c = np.log(compute)
    log_l = np.log(losses)
    # log(L) = log(A) - alpha * log(C)
    alpha, log_A = np.polyfit(log_c, log_l, deg=1)
    alpha = -alpha  # convention: L = A * C^{-alpha}, alpha > 0
    A = np.exp(log_A)
    return A, alpha


def fit_and_plot_scaling_law(
    compute_flops: List[float],
    val_losses: List[float],
    labels: List[str],
    save_path: str = "scaling_laws.png",
    colors: Optional[List[str]] = None,
    title: str = "Scaling Laws: AttnRes vs Baseline",
    figsize: tuple = (7, 5),
) -> List[tuple]:
    """Fit and plot scaling law curves for multiple model variants.

    Args:
        compute_flops: List of lists — one per variant — of compute values.
        val_losses:    List of lists — one per variant — of validation losses.
        labels:        Legend label for each variant.
        save_path:     Output file path.
        colors:        Optional list of colors, one per variant.
        title:         Plot title.
        figsize:       Matplotlib figure size.

    Returns:
        List of (A, alpha) tuples, one per variant.
    """
    assert len(compute_flops) == len(val_losses) == len(labels), \
        "compute_flops, val_losses, and labels must have the same length."

    default_colors = ["#2166ac", "#d6604d", "#4dac26", "#7b3294"]
    if colors is None:
        colors = default_colors[:len(labels)]

    fig, ax = plt.subplots(figsize=figsize)
    coeffs = []

    for c_list, l_list, label, color in zip(compute_flops, val_losses, labels, colors):
        c_arr = np.array(c_list, dtype=float)
        l_arr = np.array(l_list, dtype=float)

        # Scatter data points
        ax.scatter(c_arr, l_arr, color=color, zorder=5, s=40)

        # Fit and plot curve (need at least 2 points for a meaningful fit)
        if len(c_arr) < 2:
            coeffs.append((float("nan"), float("nan")))
            ax.scatter(c_arr, l_arr, color=color, zorder=5, s=80, label=label)
            continue
        A, alpha = fit_power_law(c_arr.tolist(), l_arr.tolist())
        coeffs.append((A, alpha))
        c_range = np.logspace(np.log10(c_arr.min()), np.log10(c_arr.max()), 200)
        l_fit = A * c_range ** (-alpha)
        ax.plot(c_range, l_fit, color=color, linewidth=2,
                label=f"{label}  (L={A:.3f}·C$^{{-{alpha:.3f}}}$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute (PFLOP/s-days)", fontsize=11)
    ax.set_ylabel("Validation Loss", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scaling law plot → {save_path}")

    return coeffs

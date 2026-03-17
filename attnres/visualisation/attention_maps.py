"""
Depth-wise attention weight visualisation (reproduces Fig 8 from arXiv:2603.15031).

For each layer l, plots the learned alpha_{i->l} weights — how much layer l
attends to each previous source. Diagonal dominance = locality preserved.
Off-diagonal concentrations = learned skip connections.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, List

from ..core.block_attn_res import BlockAttnRes
from ..core.full_attn_res import FullAttnRes
from ..core.utils import RMSNorm


def _extract_attnres(model: nn.Module):
    """Find the AttnRes module attached to a model."""
    if hasattr(model, "attnres"):
        return model.attnres
    for m in model.modules():
        if isinstance(m, (BlockAttnRes, FullAttnRes)):
            return m
    raise ValueError(
        "No AttnRes module found on model. "
        "Wrap the model with AttnResWrapper first."
    )


@torch.no_grad()
def compute_attention_weights(
    attnres,
    sample_values: Optional[List[torch.Tensor]] = None,
) -> np.ndarray:
    """Compute the depth-wise attention weight matrix alpha_{i->l}.

    For each layer l and each source i < l, computes the softmax attention
    weight that layer l assigns to source i.

    Args:
        attnres:       BlockAttnRes or FullAttnRes module.
        sample_values: Optional list of value tensors to use as keys.
                       If None, uses random unit vectors (shows structural
                       pattern from zero-init queries).

    Returns:
        weights: ndarray of shape (num_layers, max_sources).
                 weights[l, i] = alpha_{i->l}.  Entries where i >= l are NaN.
    """
    num_layers = attnres.num_layers
    d = attnres.hidden_dim
    key_norm = attnres.key_norm

    if isinstance(attnres, BlockAttnRes):
        # Determine max number of sources (N blocks + 1 for embedding)
        max_sources = attnres.num_blocks + 2
    else:
        max_sources = num_layers + 1

    weights = np.full((num_layers, max_sources), np.nan)

    # Build dummy values if not provided
    if sample_values is None:
        torch.manual_seed(0)
        sample_values = [torch.randn(d) for _ in range(num_layers + 1)]

    if isinstance(attnres, BlockAttnRes):
        # Build block representations from sample values
        block_reprs = [sample_values[0]]  # b_0 = embedding
        partial = None
        for l in range(num_layers):
            block_n = attnres._block_of(l)
            i = l - attnres._block_starts[block_n]
            fv = sample_values[l + 1]
            partial = fv if partial is None else partial + fv
            if i + 1 >= attnres.block_sizes[block_n]:
                block_reprs.append(partial)
                partial = None

        for l in range(num_layers):
            block_n = attnres._block_of(l)
            i = l - attnres._block_starts[block_n]
            n_completed = min(block_n + 1, len(block_reprs))
            sources = block_reprs[:n_completed]
            if i > 0:
                # partial sum up to layer l within block
                block_start = attnres._block_starts[block_n]
                partial_l = sum(sample_values[block_start + j + 1] for j in range(i))
                sources = sources + [partial_l]

            v_stack = torch.stack(sources)          # (S, d)
            k_stack = key_norm(v_stack)
            w = attnres.queries[l]                  # (d,)
            scores = (k_stack * w).sum(dim=-1)      # (S,)
            alpha = torch.softmax(scores, dim=0).numpy()
            weights[l, :len(alpha)] = alpha

    else:  # FullAttnRes
        all_values = sample_values[:num_layers + 1]
        for l in range(num_layers):
            sources = all_values[:l + 1]
            v_stack = torch.stack(sources)
            k_stack = key_norm(v_stack)
            w = attnres.queries[l]
            scores = (k_stack * w).sum(dim=-1)
            alpha = torch.softmax(scores, dim=0).numpy()
            weights[l, :len(alpha)] = alpha

    return weights


def plot_depth_attention(
    model: nn.Module,
    save_path: str = "depth_attention.png",
    sample_values: Optional[List[torch.Tensor]] = None,
    title: str = "Depth-wise Attention Weights α_{i→l}",
    figsize: tuple = (12, 8),
) -> np.ndarray:
    """Plot depth-wise attention weight heatmap and save to file.

    Reproduces Fig 8 of arXiv:2603.15031: each row is a consuming layer l,
    each column is a source layer i.  Colour intensity = attention weight.

    Args:
        model:         Model with attached AttnRes module.
        save_path:     Output file path (.png / .pdf).
        sample_values: Optional list of d-dim tensors to use as keys.
        title:         Plot title.
        figsize:       Matplotlib figure size.

    Returns:
        weights: The (num_layers, max_sources) weight matrix.
    """
    attnres = _extract_attnres(model)
    weights = compute_attention_weights(attnres, sample_values)

    fig, ax = plt.subplots(figsize=figsize)

    # Mask NaN entries (i >= l)
    masked = np.ma.masked_invalid(weights)
    cmap = plt.cm.Blues
    cmap.set_bad(color="#f5f5f5")

    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    plt.colorbar(im, ax=ax, label="Attention weight α_{i→l}", fraction=0.03)

    num_layers, max_sources = weights.shape
    ax.set_xlabel("Source index i")
    ax.set_ylabel("Consuming layer l")
    ax.set_title(title)

    # Tick every 4 layers for readability
    tick_step = max(1, num_layers // 16)
    ax.set_yticks(range(0, num_layers, tick_step))
    ax.set_yticklabels(range(0, num_layers, tick_step), fontsize=7)
    ax.set_xticks(range(0, max_sources, max(1, max_sources // 16)))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved depth-attention heatmap → {save_path}")

    return weights

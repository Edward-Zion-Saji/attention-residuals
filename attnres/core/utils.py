"""
Shared utilities: RMSNorm and zero-init helpers used across AttnRes modules.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no learnable parameters).

    Used to normalise keys before computing attention scores, preventing
    layers with large-magnitude outputs from dominating the softmax.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms


def zero_init_(tensor: torch.Tensor) -> torch.Tensor:
    """In-place zero-initialisation for pseudo-query parameters.

    The paper requires all pseudo-query vectors to be initialised to zero so
    that at the start of training AttnRes reduces to an equal-weight average
    over previous layer outputs, matching the standard residual baseline and
    avoiding training instability.
    """
    return nn.init.zeros_(tensor)

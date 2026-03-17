"""
AttnRes — Attention Residuals (arXiv:2603.15031).

A research implementation of Block Attention Residuals, which replaces standard
fixed residual connections with learned softmax attention over preceding layer
outputs. Models must be trained from scratch with AttnRes; it cannot be applied
as a post-hoc modification to existing pretrained checkpoints.

    from attnres.core import FullAttnRes, BlockAttnRes
    from attnres.models.gpt_demo import GPTWithAttnRes, GPTConfig
"""

__version__ = "0.1.0"

from .core import FullAttnRes, BlockAttnRes, RMSNorm

__all__ = ["FullAttnRes", "BlockAttnRes", "RMSNorm"]

"""
AttnRes — Attention Residuals library.

Implements the Attention Residuals mechanism from:
  "Attention Residuals" (Kimi Team, arXiv:2603.15031)

Usage:
    from attnres.core import FullAttnRes, BlockAttnRes
    from attnres.models.hf_wrapper import AttnResWrapper
    from attnres.inference.engine import AttnResInferenceEngine
"""

__version__ = "0.1.0"
__author__ = "AttnRes Contributors"

from .core import FullAttnRes, BlockAttnRes, RMSNorm

__all__ = ["FullAttnRes", "BlockAttnRes", "RMSNorm"]

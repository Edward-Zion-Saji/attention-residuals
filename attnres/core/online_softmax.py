"""
Online softmax merge (Milakov & Gimelshein, 2018).

Used in the two-phase Block AttnRes inference strategy to combine:
  - Phase 1 output: inter-block attention result (o1, m1, lse1)
  - Phase 2 output: intra-block attention result (o2, m2, lse2)

without re-materialising the full score vector.

Reference: https://arxiv.org/abs/1805.02867
"""

from typing import Tuple
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Low-level merge primitive
# ---------------------------------------------------------------------------

def merge_attn_stats(
    o1: torch.Tensor,
    m1: torch.Tensor,
    lse1: torch.Tensor,
    o2: torch.Tensor,
    m2: torch.Tensor,
    lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge two partial softmax attention outputs using the online softmax trick.

    Given two partial attention results with their softmax statistics (max and
    log-sum-exp), produce the combined output without re-computing the full
    softmax from scratch.

    Args:
        o1:   Partial weighted value sum from group 1.  Shape: (..., d)
        m1:   Per-element max logit from group 1.       Shape: (..., 1)
        lse1: Log-sum-exp from group 1.                 Shape: (..., 1)
        o2:   Partial weighted value sum from group 2.  Shape: (..., d)
        m2:   Per-element max logit from group 2.       Shape: (..., 1)
        lse2: Log-sum-exp from group 2.                 Shape: (..., 1)

    Returns:
        o_merged:   Combined attention output.  Shape: (..., d)
        m_merged:   Combined max.               Shape: (..., 1)
        lse_merged: Combined log-sum-exp.       Shape: (..., 1)
    """
    m_merged = torch.maximum(m1, m2)

    # Rescale each sum-exp to the common max
    e1 = torch.exp(lse1 - m_merged)  # exp(log(sum1) - m_merged) = sum1 * exp(m1 - m_merged) / exp(m1)
    e2 = torch.exp(lse2 - m_merged)
    e_total = e1 + e2

    o_merged = (o1 * e1 + o2 * e2) / e_total
    lse_merged = m_merged + torch.log(e_total)

    return o_merged, m_merged, lse_merged


# ---------------------------------------------------------------------------
# Higher-level helper that runs attention and returns stats
# ---------------------------------------------------------------------------

class AttnWithStats:
    """Compute softmax attention and return (output, max_logit, log_sum_exp).

    The raw (unnormalised) output and the softmax statistics are returned
    so that results from multiple groups can be merged via merge_attn_stats.

    All operations are performed in the calling tensor's dtype.
    """

    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run scaled dot-product attention and return (o, max, lse).

        Args:
            q: Query tensor.  Shape: (..., T_q, d)
            k: Key tensor.    Shape: (..., T_k, d)
            v: Value tensor.  Shape: (..., T_k, d)
            scale: Optional scale factor. Defaults to 1 / sqrt(d).

        Returns:
            o:   Attention output (normalised).  Shape: (..., T_q, d)
            m:   Max logit per query position.   Shape: (..., T_q, 1)
            lse: Log-sum-exp per query position. Shape: (..., T_q, 1)
        """
        d = q.shape[-1]
        if scale is None:
            scale = d ** -0.5

        # (..., T_q, T_k)
        scores = torch.matmul(q * scale, k.transpose(-2, -1))

        m = scores.max(dim=-1, keepdim=True).values          # (..., T_q, 1)
        scores_shifted = scores - m
        exp_scores = torch.exp(scores_shifted)               # (..., T_q, T_k)
        sum_exp = exp_scores.sum(dim=-1, keepdim=True)       # (..., T_q, 1)
        lse = m + torch.log(sum_exp)                         # (..., T_q, 1)

        o = torch.matmul(exp_scores / sum_exp, v)            # (..., T_q, d)

        return o, m, lse

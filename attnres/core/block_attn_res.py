"""
Block Attention Residuals (Block AttnRes).

Partitions the L layers into N blocks of S = L/N layers each.  Within each
block, layer outputs are *summed* into a single block representation b_n.
Across blocks, each layer attends over only N block-level representations
plus (within the current block) the evolving partial sum.

This reduces memory from O(L*d) to O(N*d) while recovering most of the
gain of Full AttnRes (paper shows N≈8 suffices across model scales).

Architecture (for layer i in block n, 1-indexed):
    V = [b_0, b_1, ..., b_{n-1}]           if i == 1 (first layer of block)
    V = [b_0, b_1, ..., b_{n-1}, b_n^{i-1}] if i >= 2 (intra-block partial sum)

where:
    b_0 = h_1  (token embedding, always included)
    b_n = sum_{j in B_n} f_j(h_j)          (completed block representation)
    b_n^i = partial sum over first i layers of block n

The two-phase inference strategy (Algorithm 1 from the paper):
    Phase 1: Batch ALL S pseudo-queries for block n against the N cached block
             representations → one matrix-multiply, amortised cost.
    Phase 2: Walk layers sequentially, computing intra-block attention against
             the evolving partial sum, then merge with Phase 1 via online softmax.

Reference: Section 3.2 and Section 4.2 of the AttnRes paper (arXiv:2603.15031).
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .utils import RMSNorm, zero_init_
from .online_softmax import AttnWithStats, merge_attn_stats


class BlockAttnRes(nn.Module):
    """Block Attention Residuals.

    Args:
        num_layers: Total number of transformer layers L.
        hidden_dim: Hidden dimension d.
        num_blocks:  Number of blocks N (default 8, paper recommendation).
        eps:         RMSNorm epsilon.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_blocks: int = 8,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Compute block sizes
        base_size = num_layers // num_blocks
        remainder = num_layers % num_blocks
        # Last block absorbs any remainder
        self.block_sizes = [base_size] * num_blocks
        if remainder:
            self.block_sizes[-1] += remainder
        # Cumulative layer offsets for block membership lookup
        self._block_starts = []
        offset = 0
        for s in self.block_sizes:
            self._block_starts.append(offset)
            offset += s

        # One pseudo-query per layer, zero-initialised
        self.queries = nn.Parameter(torch.empty(num_layers, hidden_dim))
        zero_init_(self.queries)

        self.key_norm = RMSNorm(hidden_dim, eps=eps)

    # ------------------------------------------------------------------
    # Helper: which block does layer l (0-based) belong to?
    # ------------------------------------------------------------------

    def _block_of(self, layer_idx: int) -> int:
        for n, start in enumerate(self._block_starts):
            if layer_idx < start + self.block_sizes[n]:
                return n
        return self.num_blocks - 1

    def _intra_block_idx(self, layer_idx: int) -> int:
        """0-based position of layer_idx within its block."""
        n = self._block_of(layer_idx)
        return layer_idx - self._block_starts[n]

    # ------------------------------------------------------------------
    # Stateful forward (used during training and sequential inference)
    # ------------------------------------------------------------------

    def reset_state(self):
        """Initialise / clear state.  Must be called before each new sequence."""
        # b_0 will be set to the token embedding
        # blocks: completed block representations [b_0, b_1, ..., b_{n-1}]
        self._blocks: List[torch.Tensor] = []
        # partial_block: running sum within the current block
        self._partial_block: Optional[torch.Tensor] = None
        self._current_block_idx: int = 0
        self._intra_idx: int = 0  # position within current block

    def set_embedding(self, embedding: torch.Tensor):
        """Set b_0 = h_1 (the token embedding).  Call after reset_state()."""
        self._blocks = [embedding]  # b_0
        self._partial_block = None
        self._current_block_idx = 0
        self._intra_idx = 0

    def push_layer_output(self, layer_out: torch.Tensor):
        """Accumulate a layer output into the current block's partial sum.

        Call this AFTER computing f_l(h_l) and BEFORE calling forward() for
        the next layer.

        Args:
            layer_out: f_l(h_l), shape (B, T, d).
        """
        if self._partial_block is None:
            self._partial_block = layer_out
        else:
            self._partial_block = self._partial_block + layer_out

        self._intra_idx += 1

        # Check if the current block is now complete
        block_n = self._current_block_idx  # 0-based block index (0 = b_0 embedding block)
        # The actual transformer block n corresponds to self._blocks having n+1 entries
        # (b_0 is always there). We are filling block (len(self._blocks) - 1), but
        # since b_0 is the embedding, the first transformer block is block index 1.
        actual_block_n = len(self._blocks) - 1  # 0-based transformer block being filled
        if actual_block_n < self.num_blocks:
            block_size = self.block_sizes[actual_block_n]
            if self._intra_idx >= block_size:
                # Block complete: store b_n and reset partial sum
                self._blocks.append(self._partial_block)
                self._partial_block = None
                self._intra_idx = 0

    def forward(self, layer_idx: int) -> torch.Tensor:
        """Compute the attention-residual input for layer `layer_idx` (0-based).

        Args:
            layer_idx: Which layer's input to compute.

        Returns:
            Tensor of shape (B, T, d).
        """
        # Determine which block and position within block
        block_n = self._block_of(layer_idx)
        i = self._intra_idx  # 0-based position within current block

        # Available block representations for inter-block attention:
        # b_0 is always available (embedding); b_1..b_{block_n-1} are completed.
        # But self._blocks = [b_0, b_1, ..., b_{block_n-1}] (up to len-1 completed)
        # However self._blocks grows as we complete blocks. For layer_idx in block_n,
        # self._blocks has exactly block_n + 1 entries (b_0 .. b_{block_n-1}, but
        # the current block is NOT yet complete, so the last entry in self._blocks
        # is b_{block_n-1} — the last completed block).
        #
        # V for the first layer of block_n (i==0):
        #   [b_0, b_1, ..., b_{block_n-1}]    (no partial sum yet)
        # V for subsequent layers (i>=1):
        #   [b_0, b_1, ..., b_{block_n-1}, partial_sum_so_far]

        # Collect value sources
        sources: List[torch.Tensor] = list(self._blocks)  # completed blocks + b_0

        if i > 0 and self._partial_block is not None:
            sources.append(self._partial_block)

        if len(sources) == 0:
            raise RuntimeError("No sources available. Did you call set_embedding()?")

        # Stack sources: (B, T, num_sources, d)
        v_stack = torch.stack(sources, dim=2)
        k_stack = self.key_norm(v_stack)

        w = self.queries[layer_idx]  # (d,)
        scores = (k_stack * w).sum(dim=-1)  # (B, T, num_sources)
        alpha = torch.softmax(scores, dim=-1)  # (B, T, num_sources)
        h = (alpha.unsqueeze(-1) * v_stack).sum(dim=2)  # (B, T, d)

        return h

    # ------------------------------------------------------------------
    # Two-phase computation (Algorithm 1 from the paper)
    # For efficient inference: amortises inter-block reads across a block.
    # ------------------------------------------------------------------

    def two_phase_block(
        self,
        block_n: int,
        layer_indices: List[int],
        block_reprs: List[torch.Tensor],
        layer_fn_outputs: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Compute all layer inputs within block_n using the two-phase strategy.

        Implements Algorithm 1 from the AttnRes paper exactly.

        Phase 1: Batch ALL S pseudo-queries for this block against the N completed
                 block representations in a single matrix multiply, recording softmax
                 statistics (max, log-sum-exp) for later online-softmax merging.
        Phase 2: Walk layers sequentially.  For the first layer (i=0), the Phase 1
                 result is used directly.  For subsequent layers (i>=1), intra-block
                 attention is computed against the accumulated partial sum b_n^{i-1}
                 and merged with the Phase 1 result via online softmax.

        Args:
            block_n:          0-based transformer-block index.
            layer_indices:    Global layer indices belonging to this block (length S).
            block_reprs:      Completed block representations seen so far:
                              [b_0, b_1, ..., b_{block_n-1}].  b_0 is the token
                              embedding and is always included.
            layer_fn_outputs: f_l(h_l) for each layer in the block, in order.
                              Required to build the partial sum during Phase 2.
                              If None, Phase 2 intra-block steps are skipped
                              (useful for pre-computing h_l before running layers).

        Returns:
            h_list:  Layer inputs for each layer. List of S tensors (B, T, d).
            partial: The accumulated b_n partial sum after all layers (B, T, d)
                     or None if layer_fn_outputs was None.
        """
        if not block_reprs:
            raise ValueError("block_reprs must contain at least b_0 (the embedding).")

        S = len(layer_indices)

        # ------------------------------------------------------------------
        # Phase 1: batched inter-block attention for all S layers at once.
        # ------------------------------------------------------------------
        # Stack completed block representations: (B, T, N_avail, d)
        v_inter = torch.stack(block_reprs, dim=2)       # (B, T, N_avail, d)
        k_inter = self.key_norm(v_inter)                 # (B, T, N_avail, d)

        B, T, N_avail, d = v_inter.shape

        # Pseudo-queries for all layers in this block: (S, d)
        W = self.queries[layer_indices]  # (S, d)

        # scores_inter[b, t, s, n] = w_s · RMSNorm(b_n)[b, t]
        # k_inter: (B, T, N_avail, d), W: (S, d)
        # → (B, T, S, N_avail)  via einsum over d
        scores_inter = torch.einsum("btnd,sd->btsn", k_inter, W)  # (B, T, S, N_avail)

        # Softmax statistics for Phase 1 — kept for online-softmax merge in Phase 2
        m1 = scores_inter.max(dim=-1, keepdim=True).values     # (B, T, S, 1)
        exp1 = torch.exp(scores_inter - m1)                     # (B, T, S, N_avail)
        sum1 = exp1.sum(dim=-1, keepdim=True)                   # (B, T, S, 1)
        lse1 = m1 + torch.log(sum1)                             # (B, T, S, 1)

        # Normalised Phase 1 output: weighted sum of values over N_avail blocks
        # o1[b, t, s, d] = sum_n (exp1[b,t,s,n]/sum1[b,t,s,1]) * v_inter[b,t,n,d]
        o1 = torch.einsum("btsn,btnd->btsd", exp1 / sum1, v_inter)  # (B, T, S, d)

        # ------------------------------------------------------------------
        # Phase 2: sequential intra-block attention + online-softmax merge.
        # ------------------------------------------------------------------
        h_list: List[torch.Tensor] = []
        partial_block: Optional[torch.Tensor] = None  # b_n^{i-1}

        for i, l in enumerate(layer_indices):
            # Slice Phase 1 result for this layer
            o1_l   = o1[:, :, i, :]    # (B, T, d)
            m1_l   = m1[:, :, i, :]    # (B, T, 1)
            lse1_l = lse1[:, :, i, :]  # (B, T, 1)

            if i == 0:
                # First layer of block: only inter-block sources → Phase 1 result.
                h_l = o1_l
            else:
                # Intra-block attention: single source = b_n^{i-1} (the partial sum).
                # Score: w_l · RMSNorm(partial_block)  per (b, t)  — scalar per token.
                k_intra = self.key_norm(partial_block)          # (B, T, d)
                w_l = self.queries[l]                            # (d,)
                score_intra = (k_intra * w_l).sum(dim=-1, keepdim=True)  # (B, T, 1)

                # For a single key, softmax is trivially 1.0 and the normalised
                # output equals the value: o2 = partial_block.
                # Stats: m2 = lse2 = score_intra  (log(exp(s)) = s for 1 key)
                m2   = score_intra   # (B, T, 1)
                lse2 = score_intra   # (B, T, 1)
                o2   = partial_block  # (B, T, d)

                # Merge Phase 1 (inter-block) and Phase 2 (intra-block) via online softmax
                h_l, _, _ = merge_attn_stats(
                    o1_l, m1_l, lse1_l,
                    o2,   m2,   lse2,
                )

            h_list.append(h_l)

            # Accumulate partial block sum for the next intra-block step
            if layer_fn_outputs is not None:
                f_out = layer_fn_outputs[i]
                if partial_block is None:
                    partial_block = f_out
                else:
                    partial_block = partial_block + f_out

        return h_list, partial_block

    # ------------------------------------------------------------------
    # Training helper: compute all layer inputs given all layer outputs
    # (batch version, no two-phase overhead needed for training)
    # ------------------------------------------------------------------

    def compute_all_inputs(
        self,
        layer_outputs: List[torch.Tensor],
        embedding: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Compute all layer inputs h_l for l=1..L given layer outputs.

        This is the straightforward training path.  For each layer, we attend
        over the appropriate block representations computed from the given
        layer outputs.

        Args:
            layer_outputs: List of L tensors of shape (B, T, d).
                           layer_outputs[l] = f_{l+1}(h_{l+1}) (0-indexed).
            embedding:     Token embedding h_1, shape (B, T, d).

        Returns:
            List of L tensors, each (B, T, d): the AttnRes input for each layer.
        """
        L = len(layer_outputs)
        inputs = []

        # Build block representations progressively
        # b_0 = embedding
        block_reprs = [embedding]   # grows as blocks complete
        partial_block: Optional[torch.Tensor] = None
        intra_idx = 0  # position within current block

        for l in range(L):
            block_n = self._block_of(l)
            i = l - self._block_starts[block_n]  # intra-block position (0-based)

            # Available value sources
            sources = list(block_reprs)
            if i > 0 and partial_block is not None:
                sources.append(partial_block)

            v_stack = torch.stack(sources, dim=2)
            k_stack = self.key_norm(v_stack)
            w = self.queries[l]
            scores = (k_stack * w).sum(dim=-1)
            alpha = torch.softmax(scores, dim=-1)
            h = (alpha.unsqueeze(-1) * v_stack).sum(dim=2)
            inputs.append(h)

            # Accumulate layer output into partial block sum
            f_l = layer_outputs[l]
            if partial_block is None:
                partial_block = f_l
            else:
                partial_block = partial_block + f_l

            # Check block completion
            if i + 1 >= self.block_sizes[block_n]:
                block_reprs.append(partial_block)
                partial_block = None

        return inputs

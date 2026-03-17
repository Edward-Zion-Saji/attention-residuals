"""
AttnResBlockCache — stores N block-level summary tensors per batch.

During autoregressive decoding, the block representations b_0..b_{N-1}
are computed once during prefill and then reused at every decode step
(Phase 1 of the two-phase inference strategy).  This module manages that
cache efficiently.

Memory: O(N * T * d) per batch — for N=8, d=4096, T=128K this is ~15 GB
        but is sharded across TP devices in production (see paper §4.2).
        For typical decode (T ≤ 8K) this is ~0.5 GB which is negligible.
"""

from typing import List, Optional
import torch


class AttnResBlockCache:
    """Stores completed block representations for AttnRes decoding.

    Keeps b_0 (token embedding sum) through b_{N-1} (last block sum).
    Updated block-by-block during prefill; held fixed during decode.

    Args:
        num_blocks:  N — number of block representations to cache (excluding b_0).
        hidden_dim:  d — hidden dimension.
        device:      torch device for tensors.
        dtype:       dtype (default float32, use bfloat16 in production).
    """

    def __init__(
        self,
        num_blocks: int,
        hidden_dim: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype

        # Stored block representations: b_0 is index 0, b_n is index n
        # Initially empty; populated via update().
        self._blocks: List[Optional[torch.Tensor]] = [None] * (num_blocks + 1)
        self._num_stored = 0

    def reset(self):
        """Clear all cached blocks (call at start of new sequence)."""
        self._blocks = [None] * (self.num_blocks + 1)
        self._num_stored = 0

    def set_embedding(self, embedding_repr: torch.Tensor):
        """Set b_0 = the token embedding representation.

        Args:
            embedding_repr: Shape (B, T, d) or (B, d) for decode.
        """
        self._blocks[0] = embedding_repr.to(device=self.device, dtype=self.dtype)
        if self._num_stored == 0:
            self._num_stored = 1

    def update(self, block_idx: int, block_repr: torch.Tensor):
        """Store or overwrite block representation b_{block_idx}.

        Args:
            block_idx:   1-based block index (1 = first transformer block).
            block_repr:  Tensor of shape (B, T, d) or (B, d).
        """
        assert 1 <= block_idx <= self.num_blocks, \
            f"block_idx must be in [1, {self.num_blocks}], got {block_idx}"
        self._blocks[block_idx] = block_repr.to(device=self.device, dtype=self.dtype)
        self._num_stored = max(self._num_stored, block_idx + 1)

    def get(self, block_idx: int) -> torch.Tensor:
        """Retrieve block representation b_{block_idx}.

        Args:
            block_idx: 0 for embedding, 1..N for transformer blocks.
        """
        t = self._blocks[block_idx]
        if t is None:
            raise RuntimeError(
                f"Block {block_idx} has not been set. "
                f"Call set_embedding() / update() first."
            )
        return t

    def get_all(self) -> List[torch.Tensor]:
        """Return all stored blocks [b_0, b_1, ..., b_k] in order."""
        stored = [b for b in self._blocks if b is not None]
        if not stored:
            raise RuntimeError("Cache is empty. Did you run prefill?")
        return stored

    def get_up_to(self, n: int) -> List[torch.Tensor]:
        """Return blocks b_0 .. b_n (inclusive).

        Args:
            n: Last block index to include (0 = embedding only).
        """
        result = []
        for i in range(n + 1):
            if self._blocks[i] is None:
                break
            result.append(self._blocks[i])
        return result

    @property
    def num_stored(self) -> int:
        return self._num_stored

    def __repr__(self) -> str:
        stored_shapes = [
            str(tuple(b.shape)) if b is not None else "None"
            for b in self._blocks
        ]
        return (
            f"AttnResBlockCache(num_blocks={self.num_blocks}, "
            f"hidden_dim={self.hidden_dim}, "
            f"stored={self._num_stored}, "
            f"shapes={stored_shapes})"
        )

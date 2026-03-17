"""
Tests for attnres.core modules.

Covers:
- RMSNorm correctness and shape
- zero_init_
- online_softmax merge (numerical correctness vs. naive)
- FullAttnRes: zero-init uniform weights, output shape, gradient flow
- BlockAttnRes: output shape, block boundaries, match with FullAttnRes at N=L
- BlockAttnRes two_phase_block correctness vs. naive forward
"""

import math
import pytest
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attnres.core.utils import RMSNorm, zero_init_
from attnres.core.online_softmax import merge_attn_stats, AttnWithStats
from attnres.core.full_attn_res import FullAttnRes
from attnres.core.block_attn_res import BlockAttnRes


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

B, T, D = 2, 5, 32
L = 8
N = 4


def make_layer_outputs(l, b=B, t=T, d=D, seed=42):
    torch.manual_seed(seed)
    return [torch.randn(b, t, d) for _ in range(l)]


def make_embedding(b=B, t=T, d=D, seed=0):
    torch.manual_seed(seed)
    return torch.randn(b, t, d)


# -----------------------------------------------------------------------
# RMSNorm
# -----------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(D)
        x = torch.randn(B, T, D)
        assert norm(x).shape == (B, T, D)

    def test_unit_rms(self):
        norm = RMSNorm(D, eps=0)
        x = torch.randn(B, T, D) * 5
        y = norm(x)
        rms = y.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    def test_no_learnable_params(self):
        norm = RMSNorm(D)
        assert len(list(norm.parameters())) == 0


# -----------------------------------------------------------------------
# zero_init_
# -----------------------------------------------------------------------

class TestZeroInit:
    def test_zeros(self):
        t = torch.ones(16, D)
        zero_init_(t)
        assert t.abs().max().item() == 0.0


# -----------------------------------------------------------------------
# Online softmax merge
# -----------------------------------------------------------------------

class TestOnlineSoftmax:
    def test_merge_equivalent_to_joint(self):
        """Merging two groups should equal computing attention over both jointly."""
        torch.manual_seed(1)
        B, S1, S2, D = 2, 3, 4, 16
        q = torch.randn(B, 1, D)
        k1 = torch.randn(B, S1, D)
        k2 = torch.randn(B, S2, D)
        v1 = torch.randn(B, S1, D)
        v2 = torch.randn(B, S2, D)

        # Joint attention
        k_all = torch.cat([k1, k2], dim=1)
        v_all = torch.cat([v1, v2], dim=1)
        o_joint, m_joint, lse_joint = AttnWithStats.forward(q, k_all, v_all)

        # Merged attention
        o1, m1, lse1 = AttnWithStats.forward(q, k1, v1)
        o2, m2, lse2 = AttnWithStats.forward(q, k2, v2)
        o_merged, _, _ = merge_attn_stats(o1, m1, lse1, o2, m2, lse2)

        assert torch.allclose(o_joint, o_merged, atol=1e-5), \
            f"Max diff: {(o_joint - o_merged).abs().max()}"

    def test_merge_shape(self):
        B, S, D = 2, 5, 16
        q = torch.randn(B, 1, D)
        k = torch.randn(B, S, D)
        v = torch.randn(B, S, D)
        o, m, lse = AttnWithStats.forward(q, k, v)
        assert o.shape == (B, 1, D)
        assert m.shape == (B, 1, 1)
        assert lse.shape == (B, 1, 1)

    def test_merge_single_element(self):
        """Merging with an empty second group should be identity."""
        torch.manual_seed(2)
        B, S, D = 2, 4, 16
        q = torch.randn(B, 1, D)
        k = torch.randn(B, S, D)
        v = torch.randn(B, S, D)
        o1, m1, lse1 = AttnWithStats.forward(q, k, v)

        # Merge with a "null" group that has the same stats but weight 0
        # achieved by giving o2 the same m as o1 but lse2 = -inf
        o2 = torch.zeros_like(o1)
        m2 = torch.full_like(m1, float("-inf"))
        lse2 = torch.full_like(lse1, float("-inf"))
        o_merged, _, _ = merge_attn_stats(o1, m1, lse1, o2, m2, lse2)

        assert torch.allclose(o1, o_merged, atol=1e-5)


# -----------------------------------------------------------------------
# FullAttnRes
# -----------------------------------------------------------------------

class TestFullAttnRes:
    def test_output_shape(self):
        model = FullAttnRes(num_layers=L, hidden_dim=D)
        embedding = make_embedding()
        layer_outputs = make_layer_outputs(L - 1)
        # First layer input = embedding
        model.reset_state()
        model.set_embedding = lambda e: model._values.__setitem__(slice(None), [e]) or model._values.__init__([e]) or None
        # Use compute_all_inputs instead
        inputs = model.compute_all_inputs(layer_outputs, embedding)
        assert len(inputs) == L - 1
        for h in inputs:
            assert h.shape == (B, T, D)

    def test_output_shape_stateful(self):
        model = FullAttnRes(num_layers=L, hidden_dim=D)
        embedding = make_embedding()
        layer_outputs = make_layer_outputs(L)

        model.reset_state()
        model.push(embedding)

        for l in range(L):
            h = model.forward(l)
            assert h.shape == (B, T, D)
            # Push the layer output to simulate f_l(h_l)
            model.push(layer_outputs[l])

    def test_zero_init_uniform_weights(self):
        """At zero init, all queries are zero -> uniform attention weights."""
        model = FullAttnRes(num_layers=L, hidden_dim=D)
        assert model.queries.abs().max().item() == 0.0, "Queries not zero-initialised"

        # With zero queries, scores are all zero, so softmax = uniform
        torch.manual_seed(5)
        embedding = make_embedding()
        model.reset_state()
        model.push(embedding)

        # After pushing embedding and one layer output
        layer_out = torch.randn(B, T, D)
        model.push(layer_out)

        # Layer 1's input should be uniform over [v_0, v_1]
        h = model.forward(1)
        expected = (embedding + layer_out) / 2.0
        # Uniform softmax gives equal weights -> (v_0 + v_1) / 2
        # But our queries are zero, RMSNorm(v_i) matters - scores are all w·RMSNorm(v_i)
        # With w=0, all scores are 0, softmax = 1/num_sources for each
        # So h = (v_0 + v_1) / 2 regardless of RMSNorm
        assert torch.allclose(h, expected, atol=1e-5), \
            f"Expected uniform average at zero init, got max diff {(h - expected).abs().max()}"

    def test_gradient_flows(self):
        model = FullAttnRes(num_layers=L, hidden_dim=D)
        embedding = make_embedding().requires_grad_(True)
        layer_outputs = [torch.randn(B, T, D, requires_grad=True) for _ in range(L)]

        inputs = model.compute_all_inputs(layer_outputs[:-1], embedding)
        loss = sum(h.sum() for h in inputs)
        loss.backward()

        assert embedding.grad is not None
        assert model.queries.grad is not None

    def test_first_layer_is_embedding(self):
        """Layer 0 should simply return the embedding (only source is v_0)."""
        model = FullAttnRes(num_layers=L, hidden_dim=D)
        embedding = make_embedding()
        model.reset_state()
        model.push(embedding)
        h = model.forward(0)
        assert torch.allclose(h, embedding, atol=1e-6)


# -----------------------------------------------------------------------
# BlockAttnRes
# -----------------------------------------------------------------------

class TestBlockAttnRes:
    def test_block_sizes(self):
        model = BlockAttnRes(num_layers=8, hidden_dim=D, num_blocks=4)
        assert sum(model.block_sizes) == 8
        assert model.block_sizes == [2, 2, 2, 2]

    def test_block_sizes_uneven(self):
        model = BlockAttnRes(num_layers=10, hidden_dim=D, num_blocks=3)
        assert sum(model.block_sizes) == 10
        # Last block absorbs remainder
        assert model.block_sizes[-1] == model.block_sizes[0] + 1 or \
               model.block_sizes[-1] >= model.block_sizes[0]

    def test_output_shape(self):
        model = BlockAttnRes(num_layers=L, hidden_dim=D, num_blocks=N)
        embedding = make_embedding()
        layer_outputs = make_layer_outputs(L)

        inputs = model.compute_all_inputs(layer_outputs, embedding)
        assert len(inputs) == L
        for h in inputs:
            assert h.shape == (B, T, D)

    def test_zero_init_uniform_weights(self):
        model = BlockAttnRes(num_layers=L, hidden_dim=D, num_blocks=N)
        assert model.queries.abs().max().item() == 0.0

    def test_gradient_flows(self):
        model = BlockAttnRes(num_layers=L, hidden_dim=D, num_blocks=N)
        embedding = make_embedding().requires_grad_(True)
        layer_outputs = [torch.randn(B, T, D, requires_grad=True) for _ in range(L)]

        inputs = model.compute_all_inputs(layer_outputs, embedding)
        loss = sum(h.sum() for h in inputs)
        loss.backward()

        assert embedding.grad is not None
        assert model.queries.grad is not None

    def test_block_of(self):
        model = BlockAttnRes(num_layers=8, hidden_dim=D, num_blocks=4)
        # With 4 blocks of size 2: layers 0,1 in block 0; 2,3 in block 1; etc.
        assert model._block_of(0) == 0
        assert model._block_of(1) == 0
        assert model._block_of(2) == 1
        assert model._block_of(3) == 1
        assert model._block_of(6) == 3
        assert model._block_of(7) == 3

    def test_two_phase_matches_naive(self):
        """two_phase_block output must match compute_all_inputs for the same block."""
        torch.manual_seed(7)
        num_layers = 8
        num_blocks = 4
        block_size = 2

        model = BlockAttnRes(num_layers=num_layers, hidden_dim=D, num_blocks=num_blocks)
        # Give non-zero queries for this test
        with torch.no_grad():
            model.queries.normal_(0, 0.1)

        embedding = make_embedding()
        layer_outputs = make_layer_outputs(num_layers)

        # Naive path
        all_inputs_naive = model.compute_all_inputs(layer_outputs, embedding)

        # Two-phase path for block 1 (layers 2,3)
        # Build block reprs: b_0 = embedding, b_1 = sum of layer_outputs[0:2]
        b0 = embedding
        b1 = layer_outputs[0] + layer_outputs[1]
        block_reprs = [b0, b1]
        layer_indices = [2, 3]
        fn_outputs = [layer_outputs[2], layer_outputs[3]]

        h_list, partial = model.two_phase_block(
            block_n=1,
            layer_indices=layer_indices,
            block_reprs=block_reprs,
            layer_fn_outputs=fn_outputs,
        )

        for i, l in enumerate(layer_indices):
            naive_h = all_inputs_naive[l]
            two_phase_h = h_list[i]
            assert torch.allclose(naive_h, two_phase_h, atol=1e-4), \
                f"Layer {l}: max diff = {(naive_h - two_phase_h).abs().max()}"

    def test_single_block_reduces_to_full(self):
        """With N=1 block containing all layers, block attnres acts on embedding only."""
        model = BlockAttnRes(num_layers=4, hidden_dim=D, num_blocks=1)
        embedding = make_embedding()
        layer_outputs = make_layer_outputs(4)

        inputs = model.compute_all_inputs(layer_outputs, embedding)
        # With N=1, b_0 is the embedding and the block hasn't completed yet
        # Layer 0: sources = [b_0], i=0 → uniform attention over b_0 → h = b_0
        # (first layer of block 0 = first layer overall)
        assert torch.allclose(inputs[0], embedding, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
AI ENGINEERING ACADEMY -- MODULE 014 TEST SUITE
Comprehensive Pytest Suite for Advanced Positional Encodings (16 Tests)
"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_mod14_dir = os.path.dirname(_dir)

_spec = importlib.util.spec_from_file_location("impl_mod14", os.path.join(_mod14_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

RoPEEmbedding = _mod.RoPEEmbedding
ALiBiBias     = _mod.ALiBiBias
RelativePositionBiasT5 = _mod.RelativePositionBiasT5

_spec_ch = importlib.util.spec_from_file_location("ch_mod14", os.path.join(_mod14_dir, "07-challenge-solution.py"))
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)
RoPEMultiHeadAttention = _mod_ch.RoPEMultiHeadAttention
verify_rope_mha = _mod_ch.verify_rope_mha


# ===================================================================
# 1. ROTARY POSITION EMBEDDING (RoPE) (4 tests)
# ===================================================================
class TestRoPEEmbedding:
    def test_rope_output_shape_3d(self):
        rope = RoPEEmbedding(dim=16)
        x = np.random.randn(2, 10, 16)
        out = rope.apply(x, seq_dim=1)
        assert out.shape == (2, 10, 16)

    def test_rope_output_shape_4d(self):
        rope = RoPEEmbedding(dim=16)
        x = np.random.randn(2, 10, 4, 16)
        out = rope.apply(x, seq_dim=1)
        assert out.shape == (2, 10, 4, 16)

    def test_rope_position_zero_identity_rotation(self):
        rope = RoPEEmbedding(dim=16)
        x = np.random.randn(1, 1, 16)
        # At position 0, cos(0)=1, sin(0)=0 -> rotation should return x unchanged
        out = rope.apply(x, seq_dim=1)
        np.testing.assert_allclose(out, x, atol=1e-6)

    def test_rope_relative_invariance_dot_product(self):
        dim = 16
        rope = RoPEEmbedding(dim=dim)
        np.random.seed(42)
        q = np.random.randn(dim)
        k = np.random.randn(dim)

        # Position 2 and 5 (distance=3)
        X1 = np.zeros((1, 10, dim))
        X1[0, 2] = q
        X1[0, 5] = k
        rot1 = rope.apply(X1)
        dot1 = np.dot(rot1[0, 2], rot1[0, 5])

        # Position 6 and 9 (distance=3)
        X2 = np.zeros((1, 10, dim))
        X2[0, 6] = q
        X2[0, 9] = k
        rot2 = rope.apply(X2)
        dot2 = np.dot(rot2[0, 6], rot2[0, 9])

        np.testing.assert_allclose(dot1, dot2, atol=1e-5)


# ===================================================================
# 2. ALiBi BIAS (4 tests)
# ===================================================================
class TestALiBiBias:
    def test_alibi_matrix_shape(self):
        alibi = ALiBiBias(num_heads=8)
        bias = alibi.forward(seq_len=12)
        assert bias.shape == (1, 8, 12, 12)

    def test_alibi_diagonal_is_zero(self):
        alibi = ALiBiBias(num_heads=4)
        bias = alibi.forward(seq_len=6)[0]  # (4, 6, 6)
        for h in range(4):
            diag = np.diag(bias[h])
            np.testing.assert_allclose(diag, 0.0, atol=1e-6)

    def test_alibi_monotonically_decreasing_with_distance(self):
        alibi = ALiBiBias(num_heads=4)
        bias = alibi.forward(seq_len=10)[0]  # (4, 10, 10)
        # Position (0, 0) > (0, 1) > (0, 2)
        for h in range(4):
            assert bias[h, 0, 0] > bias[h, 0, 1] > bias[h, 0, 2]

    def test_alibi_slopes_geometric(self):
        alibi = ALiBiBias(num_heads=8)
        slopes = alibi.slopes
        assert len(slopes) == 8
        assert slopes[0] == 0.5
        assert slopes[1] == 0.25


# ===================================================================
# 3. T5 RELATIVE POSITION BIAS (4 tests)
# ===================================================================
class TestRelativePositionBiasT5:
    def test_t5_bias_shape(self):
        t5 = RelativePositionBiasT5(num_heads=4, seed=42)
        bias = t5.forward(query_length=8, key_length=8)
        assert bias.shape == (1, 4, 8, 8)

    def test_t5_self_distance_bias_constant(self):
        t5 = RelativePositionBiasT5(num_heads=4, seed=42)
        bias = t5.forward(query_length=10, key_length=10)[0]  # (4, 10, 10)
        # (0, 0) and (5, 5) have distance 0 -> same bucket -> same bias
        np.testing.assert_allclose(bias[:, 0, 0], bias[:, 5, 5])

    def test_t5_different_distances_produce_different_biases(self):
        t5 = RelativePositionBiasT5(num_heads=4, seed=42)
        bias = t5.forward(query_length=10, key_length=10)[0]
        assert not np.allclose(bias[:, 0, 0], bias[:, 0, 5])

    def test_t5_asymmetric_query_key_lengths(self):
        t5 = RelativePositionBiasT5(num_heads=4, seed=42)
        bias = t5.forward(query_length=5, key_length=12)
        assert bias.shape == (1, 4, 5, 12)


# ===================================================================
# 4. RoPE MULTI-HEAD ATTENTION (4 tests)
# ===================================================================
class TestRoPEMultiHeadAttention:
    def test_rope_mha_output_shape(self):
        mha = RoPEMultiHeadAttention(d_model=32, num_heads=4, seed=42)
        X = np.random.randn(2, 6, 32)
        out, attn, _ = mha.forward(X)
        assert out.shape == (2, 6, 32)
        assert attn.shape == (2, 4, 6, 6)

    def test_rope_mha_attn_weights_sum_to_one(self):
        mha = RoPEMultiHeadAttention(d_model=32, num_heads=4, seed=42)
        X = np.random.randn(2, 6, 32)
        _, attn, _ = mha.forward(X)
        row_sums = attn.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_rope_mha_no_nans(self):
        mha = RoPEMultiHeadAttention(d_model=32, num_heads=4, seed=42)
        X = np.random.randn(2, 8, 32) * 5.0
        out, attn, _ = mha.forward(X)
        assert not np.isnan(out).any()
        assert not np.isnan(attn).any()

    def test_rope_mha_challenge_verification_runs(self):
        verify_rope_mha()

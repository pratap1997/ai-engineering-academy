"""
AI ENGINEERING ACADEMY — MODULE 010 TEST SUITE
Comprehensive Pytest Suite for Attention Mechanisms (16 Tests)
"""

import importlib.util
import os
import numpy as np
import pytest

_script_dir = os.path.dirname(os.path.abspath(__file__))
_mod10_dir = os.path.dirname(_script_dir)
_spec = importlib.util.spec_from_file_location(
    "implementation_mod10",
    os.path.join(_mod10_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ScaledDotProductAttention = _mod.ScaledDotProductAttention
MultiHeadAttention = _mod.MultiHeadAttention
SinusoidalPositionalEncoding = _mod.SinusoidalPositionalEncoding
softmax = _mod.softmax

_spec_ch = importlib.util.spec_from_file_location(
    "challenge_mod10",
    os.path.join(_mod10_dir, "07-challenge-solution.py"),
)
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)
verify_attention_gradcheck = _mod_ch.verify_attention_gradcheck


# =====================================================================
# 1. SCALED DOT-PRODUCT ATTENTION (4 tests)
# =====================================================================
class TestScaledDotProductAttention:
    def test_sdpa_output_shape(self):
        sdpa = ScaledDotProductAttention()
        Q = np.random.randn(2, 5, 16)
        K = np.random.randn(2, 5, 16)
        V = np.random.randn(2, 5, 32)
        out, attn = sdpa.forward(Q, K, V)
        assert out.shape == (2, 5, 32)
        assert attn.shape == (2, 5, 5)

    def test_sdpa_attention_weights_sum_to_one(self):
        sdpa = ScaledDotProductAttention()
        Q = np.random.randn(3, 4, 8)
        K = np.random.randn(3, 4, 8)
        V = np.random.randn(3, 4, 8)
        _, attn = sdpa.forward(Q, K, V)
        row_sums = attn.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_sdpa_attention_weights_nonnegative(self):
        sdpa = ScaledDotProductAttention()
        Q = np.random.randn(1, 6, 4)
        K = np.random.randn(1, 6, 4)
        V = np.random.randn(1, 6, 4)
        _, attn = sdpa.forward(Q, K, V)
        assert np.all(attn >= 0.0)

    def test_sdpa_causal_mask_zeros_future_attention(self):
        sdpa = ScaledDotProductAttention()
        T = 4
        Q = np.random.randn(1, T, 4)
        K = np.random.randn(1, T, 4)
        V = np.random.randn(1, T, 4)
        # Causal mask: upper triangle (future positions)
        mask = np.triu(np.ones((1, T, T)), k=1)
        _, attn = sdpa.forward(Q, K, V, mask=mask)
        # Upper triangle should be ~0
        for i in range(T):
            for j in range(i + 1, T):
                assert attn[0, i, j] < 1e-5


# =====================================================================
# 2. MULTI-HEAD ATTENTION (4 tests)
# =====================================================================
class TestMultiHeadAttention:
    def test_mha_output_shape(self):
        mha = MultiHeadAttention(d_model=32, num_heads=4, seed=42)
        X = np.random.randn(2, 6, 32)
        out, attn = mha.forward(X, X, X)
        assert out.shape == (2, 6, 32)

    def test_mha_attention_heads_count(self):
        H = 4
        mha = MultiHeadAttention(d_model=32, num_heads=H, seed=42)
        X = np.random.randn(2, 5, 32)
        _, attn = mha.forward(X, X, X)
        assert attn.shape[1] == H

    def test_mha_weight_matrices_correct_shape(self):
        mha = MultiHeadAttention(d_model=64, num_heads=8, seed=42)
        assert mha.W_Q.shape == (64, 64)
        assert mha.W_K.shape == (64, 64)
        assert mha.W_V.shape == (64, 64)
        assert mha.W_O.shape == (64, 64)

    def test_mha_parameter_count_equals_4x_dmodel_squared(self):
        d_model = 32
        mha = MultiHeadAttention(d_model=d_model, num_heads=4, seed=42)
        total_params = (mha.W_Q.size + mha.W_K.size + mha.W_V.size + mha.W_O.size)
        assert total_params == 4 * d_model * d_model


# =====================================================================
# 3. POSITIONAL ENCODING (4 tests)
# =====================================================================
class TestSinusoidalPositionalEncoding:
    def test_pe_output_shape_matches_input(self):
        pe = SinusoidalPositionalEncoding(d_model=32, max_len=100)
        X = np.random.randn(2, 10, 32)
        out = pe.forward(X)
        assert out.shape == X.shape

    def test_pe_position_zero_sin_equals_zero(self):
        pe = SinusoidalPositionalEncoding(d_model=16, max_len=50)
        # PE[0, 0] = sin(0) = 0.0
        assert abs(pe.pe[0, 0] - 0.0) < 1e-6

    def test_pe_position_zero_cos_equals_one(self):
        pe = SinusoidalPositionalEncoding(d_model=16, max_len=50)
        # PE[0, 1] = cos(0) = 1.0
        assert abs(pe.pe[0, 1] - 1.0) < 1e-6

    def test_pe_different_positions_have_different_encodings(self):
        pe = SinusoidalPositionalEncoding(d_model=32, max_len=50)
        assert not np.allclose(pe.pe[0], pe.pe[1])
        assert not np.allclose(pe.pe[1], pe.pe[10])


# =====================================================================
# 4. END-TO-END ATTENTION (4 tests)
# =====================================================================
class TestEndToEndAttention:
    def test_sqrt_dk_scaling_reduces_large_scores(self):
        np.random.seed(42)
        d_k = 512
        q = np.random.randn(d_k)
        k = np.random.randn(d_k)
        raw = q @ k
        scaled = raw / np.sqrt(d_k)
        assert abs(scaled) < abs(raw)

    def test_attention_challenge_gradcheck(self):
        verify_attention_gradcheck()

    def test_mha_self_attention_no_nans(self):
        mha = MultiHeadAttention(d_model=32, num_heads=4, seed=42)
        X = np.random.randn(2, 8, 32) * 10.0
        out, _ = mha.forward(X, X, X)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_mha_cross_attention_different_seq_lengths(self):
        mha = MultiHeadAttention(d_model=32, num_heads=4, seed=42)
        Q = np.random.randn(2, 5, 32)  # Decoder: T_q=5
        K = np.random.randn(2, 8, 32)  # Encoder: T_k=8
        V = np.random.randn(2, 8, 32)
        out, attn = mha.forward(Q, K, V)
        assert out.shape == (2, 5, 32)
        assert attn.shape == (2, 4, 5, 8)

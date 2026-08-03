"""
AI ENGINEERING ACADEMY — MODULE 011 TEST SUITE
Comprehensive Pytest Suite for Transformer Block (16 Tests)
"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_mod11_dir = os.path.dirname(_dir)

_spec = importlib.util.spec_from_file_location("impl_mod11", os.path.join(_mod11_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

LayerNorm = _mod.LayerNorm
TransformerFFN = _mod.TransformerFFN
TransformerEncoderBlock = _mod.TransformerEncoderBlock
TransformerEncoder = _mod.TransformerEncoder
gelu = _mod.gelu

_spec_ch = importlib.util.spec_from_file_location("challenge_mod11", os.path.join(_mod11_dir, "07-challenge-solution.py"))
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)
verify_encoder_gradcheck = _mod_ch.verify_encoder_gradcheck


# =====================================================================
# 1. LAYER NORMALIZATION (4 tests)
# =====================================================================
class TestLayerNorm:
    def test_layernorm_zero_mean_per_token(self):
        ln = LayerNorm(32)
        x = np.random.randn(2, 5, 32) * 100.0
        out = ln.forward(x)
        means = out.mean(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)

    def test_layernorm_unit_variance_per_token(self):
        ln = LayerNorm(64)
        x = np.random.randn(2, 6, 64) * 50.0
        out = ln.forward(x)
        # With gamma=1, beta=0, var should be ~1 (within eps tolerance)
        vars_ = out.var(axis=-1)
        np.testing.assert_allclose(vars_, 1.0, atol=2e-2)

    def test_layernorm_output_shape_unchanged(self):
        ln = LayerNorm(16)
        x = np.random.randn(3, 7, 16)
        assert ln.forward(x).shape == (3, 7, 16)

    def test_layernorm_gamma_beta_learnable_initialized(self):
        ln = LayerNorm(32)
        np.testing.assert_allclose(ln.gamma, np.ones(32))
        np.testing.assert_allclose(ln.beta, np.zeros(32))


# =====================================================================
# 2. TRANSFORMER FFN (4 tests)
# =====================================================================
class TestTransformerFFN:
    def test_ffn_output_shape(self):
        ffn = TransformerFFN(d_model=32, seed=42)
        x = np.random.randn(2, 5, 32)
        assert ffn.forward(x).shape == (2, 5, 32)

    def test_ffn_expansion_ratio_is_4x(self):
        d_model = 64
        ffn = TransformerFFN(d_model=d_model, seed=42)
        assert ffn.d_ff == 4 * d_model

    def test_ffn_no_nans_with_large_input(self):
        ffn = TransformerFFN(d_model=32, seed=42)
        x = np.random.randn(2, 5, 32) * 100.0
        out = ffn.forward(x)
        assert not np.isnan(out).any()

    def test_gelu_positive_region_approximately_linear(self):
        x = np.array([10.0])
        assert abs(gelu(x)[0] - 10.0) < 0.01

    def test_gelu_negative_region_near_zero(self):
        x = np.array([-10.0])
        assert abs(gelu(x)[0]) < 0.01


# =====================================================================
# 3. TRANSFORMER ENCODER BLOCK (4 tests)
# =====================================================================
class TestTransformerEncoderBlock:
    def test_block_output_shape(self):
        block = TransformerEncoderBlock(d_model=32, num_heads=4, seed=42)
        x = np.random.randn(2, 6, 32)
        assert block.forward(x).shape == (2, 6, 32)

    def test_block_residual_connection_not_zero(self):
        block = TransformerEncoderBlock(d_model=32, num_heads=4, seed=42)
        x = np.random.randn(1, 4, 32)
        out = block.forward(x)
        # Residual means output != attention output alone
        assert not np.allclose(x, out)

    def test_block_no_nans(self):
        block = TransformerEncoderBlock(d_model=32, num_heads=4, seed=42)
        x = np.random.randn(2, 8, 32) * 5.0
        assert not np.isnan(block.forward(x)).any()

    def test_encoder_gradcheck(self):
        verify_encoder_gradcheck()


# =====================================================================
# 4. STACKED TRANSFORMER ENCODER (4 tests)
# =====================================================================
class TestTransformerEncoder:
    def test_stacked_encoder_output_shape(self):
        enc = TransformerEncoder(d_model=32, num_heads=4, num_layers=3, seed=42)
        x = np.random.randn(2, 8, 32)
        assert enc.forward(x).shape == (2, 8, 32)

    def test_stacked_encoder_correct_num_blocks(self):
        enc = TransformerEncoder(d_model=32, num_heads=4, num_layers=5, seed=42)
        assert len(enc.blocks) == 5

    def test_stacked_encoder_no_nans_6_layers(self):
        enc = TransformerEncoder(d_model=64, num_heads=4, num_layers=6, seed=42)
        x = np.random.randn(2, 10, 64)
        assert not np.isnan(enc.forward(x)).any()

    def test_stacked_encoder_different_input_produces_different_output(self):
        enc = TransformerEncoder(d_model=32, num_heads=4, num_layers=2, seed=42)
        x1 = np.random.randn(1, 5, 32)
        x2 = np.random.randn(1, 5, 32)
        assert not np.allclose(enc.forward(x1), enc.forward(x2))

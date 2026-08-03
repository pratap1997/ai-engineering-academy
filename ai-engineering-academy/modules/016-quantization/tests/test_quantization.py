"""
AI ENGINEERING ACADEMY -- MODULE 016 TEST SUITE
Comprehensive Pytest Suite for Model Quantization (16 Tests)
"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_mod16_dir = os.path.dirname(_dir)

_spec = importlib.util.spec_from_file_location("impl_mod16", os.path.join(_mod16_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SymmetricQuantizer  = _mod.SymmetricQuantizer
AsymmetricQuantizer = _mod.AsymmetricQuantizer
GroupQuantizer      = _mod.GroupQuantizer
QuantizedLinear     = _mod.QuantizedLinear

_spec_ch = importlib.util.spec_from_file_location("ch_mod16", os.path.join(_mod16_dir, "07-challenge-solution.py"))
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)
QuantizedMultiHeadAttention = _mod_ch.QuantizedMultiHeadAttention
verify_quantized_mha = _mod_ch.verify_quantized_mha


# ===================================================================
# 1. SYMMETRIC QUANTIZER (4 tests)
# ===================================================================
class TestSymmetricQuantizer:
    def test_sym8_bounds(self):
        sq = SymmetricQuantizer(bits=8)
        x = np.array([-10.0, 0.0, 5.0, 10.0])
        q, s = sq.quantize(x)
        assert q.min() >= -127
        assert q.max() <= 127

    def test_sym4_bounds(self):
        sq = SymmetricQuantizer(bits=4)
        x = np.array([-5.0, 0.0, 5.0])
        q, s = sq.quantize(x)
        assert q.min() >= -7
        assert q.max() <= 7

    def test_sym8_reconstruction_error_low(self):
        sq = SymmetricQuantizer(bits=8)
        x = np.random.randn(100) * 3.0
        q, s = sq.quantize(x)
        x_hat = sq.dequantize(q, s)
        mse = np.mean((x - x_hat) ** 2)
        assert mse < 1e-2

    def test_sym8_zero_input(self):
        sq = SymmetricQuantizer(bits=8)
        x = np.zeros(10)
        q, s = sq.quantize(x)
        assert (q == 0).all()


# ===================================================================
# 2. ASYMMETRIC QUANTIZER (4 tests)
# ===================================================================
class TestAsymmetricQuantizer:
    def test_asym8_bounds(self):
        asym = AsymmetricQuantizer(bits=8)
        x = np.random.randn(50) * 10.0
        q, s, z = asym.quantize(x)
        assert q.min() >= 0
        assert q.max() <= 255

    def test_asym8_zero_point_within_range(self):
        asym = AsymmetricQuantizer(bits=8)
        x = np.array([-5.0, 15.0])
        _, _, z = asym.quantize(x)
        assert 0 <= z <= 255

    def test_asym8_reconstruction_error_low(self):
        asym = AsymmetricQuantizer(bits=8)
        x = np.random.randn(100) * 5.0
        q, s, z = asym.quantize(x)
        x_hat = asym.dequantize(q, s, z)
        mse = np.mean((x - x_hat) ** 2)
        assert mse < 1e-2

    def test_asym4_bounds(self):
        asym = AsymmetricQuantizer(bits=4)
        x = np.array([0.0, 10.0])
        q, s, z = asym.quantize(x)
        assert q.min() >= 0
        assert q.max() <= 15


# ===================================================================
# 3. GROUP QUANTIZER & QUANTIZED LINEAR (4 tests)
# ===================================================================
class TestGroupQuantizerAndLinear:
    def test_group_quantizer_shapes(self):
        gq = GroupQuantizer(group_size=64, bits=4)
        W = np.random.randn(128, 256)
        Q_W, scales = gq.quantize(W)
        assert Q_W.shape == (128, 256)
        assert scales.shape == (128, 4)  # 256 / 64 = 4 groups

    def test_group_quantizer_reconstruction_mse(self):
        gq = GroupQuantizer(group_size=64, bits=4)
        W = np.random.randn(64, 128)
        Q_W, scales = gq.quantize(W)
        W_hat = gq.dequantize(Q_W, scales)
        mse = np.mean((W - W_hat) ** 2)
        assert mse < 0.1

    def test_quantized_linear_forward_shape(self):
        ql = QuantizedLinear(in_features=128, out_features=64, bits=4, group_size=64, seed=42)
        x = np.random.randn(3, 128)
        out = ql.forward(x)
        assert out.shape == (3, 64)

    def test_quantized_linear_no_nans(self):
        ql = QuantizedLinear(in_features=64, out_features=32, bits=4, group_size=32, seed=42)
        x = np.random.randn(2, 64) * 10.0
        out = ql.forward(x)
        assert not np.isnan(out).any()


# ===================================================================
# 4. QUANTIZED ATTENTION & CHALLENGE (4 tests)
# ===================================================================
class TestQuantizedAttention:
    def test_quantized_mha_forward_shape(self):
        qmha = QuantizedMultiHeadAttention(d_model=32, num_heads=4, bits=4, group_size=32, seed=42)
        x = np.random.randn(2, 6, 32)
        out, attn = qmha.forward(x)
        assert out.shape == (2, 6, 32)
        assert attn.shape == (2, 4, 6, 6)

    def test_quantized_mha_attn_sum_to_one(self):
        qmha = QuantizedMultiHeadAttention(d_model=32, num_heads=4, bits=4, group_size=32, seed=42)
        x = np.random.randn(2, 5, 32)
        _, attn = qmha.forward(x)
        row_sums = attn.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_quantized_mha_no_nans(self):
        qmha = QuantizedMultiHeadAttention(d_model=32, num_heads=4, bits=4, group_size=32, seed=42)
        x = np.random.randn(1, 4, 32)
        out, attn = qmha.forward(x)
        assert not np.isnan(out).any()

    def test_challenge_verification_runs(self):
        verify_quantized_mha()

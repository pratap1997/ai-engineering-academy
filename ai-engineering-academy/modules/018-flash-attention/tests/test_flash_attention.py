"""
AI ENGINEERING ACADEMY -- MODULE 018 TEST SUITE
Comprehensive Pytest Suite for FlashAttention & Online Softmax (16 Tests)
"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_mod18_dir = os.path.dirname(_dir)

_spec = importlib.util.spec_from_file_location("impl_mod18", os.path.join(_mod18_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

OnlineSoftmax       = _mod.OnlineSoftmax
FlashAttentionTiled = _mod.FlashAttentionTiled
standard_attention  = _mod.standard_attention

_spec_ch = importlib.util.spec_from_file_location("ch_mod18", os.path.join(_mod18_dir, "07-challenge-solution.py"))
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)
CausalFlashAttention = _mod_ch.CausalFlashAttention
verify_causal_flash_attention = _mod_ch.verify_causal_flash_attention


# ===================================================================
# 1. ONLINE SOFTMAX (4 tests)
# ===================================================================
class TestOnlineSoftmax:
    def test_online_softmax_first_update(self):
        acc = OnlineSoftmax()
        x1 = np.array([[1.0, 2.0, 3.0]])
        p1, m1, d1 = acc.update(x1)
        assert m1[0, 0] == 3.0
        assert abs(d1[0, 0] - np.sum(np.exp(x1 - 3.0))) < 1e-6

    def test_online_softmax_consecutive_updates(self):
        acc = OnlineSoftmax()
        x1 = np.array([[1.0, 2.0]])
        acc.update(x1)
        x2 = np.array([[4.0, 0.0]])
        _, m2, d2 = acc.update(x2)
        assert m2[0, 0] == 4.0

    def test_online_softmax_matches_full_softmax(self):
        x = np.random.randn(1, 10)
        full_max = np.max(x)
        full_exp = np.exp(x - full_max)
        full_sum = np.sum(full_exp)

        acc = OnlineSoftmax()
        acc.update(x[:, :5])
        _, m_final, d_final = acc.update(x[:, 5:])

        np.testing.assert_allclose(m_final[0, 0], full_max, atol=1e-6)
        np.testing.assert_allclose(d_final[0, 0], full_sum, atol=1e-6)

    def test_online_softmax_single_element(self):
        acc = OnlineSoftmax()
        p, m, d = acc.update(np.array([[5.0]]))
        assert m[0, 0] == 5.0
        assert d[0, 0] == 1.0


# ===================================================================
# 2. FLASHATTENTION TILED FORWARD PASS (4 tests)
# ===================================================================
class TestFlashAttentionTiled:
    def test_flash_attention_output_shape(self):
        flash = FlashAttentionTiled(block_r=16, block_c=16)
        Q = np.random.randn(2, 4, 32, 16)
        K = np.random.randn(2, 4, 32, 16)
        V = np.random.randn(2, 4, 32, 16)
        out = flash.forward(Q, K, V)
        assert out.shape == (2, 4, 32, 16)

    def test_flash_attention_exact_numerical_equivalence(self):
        np.random.seed(42)
        Q = np.random.randn(1, 2, 48, 16)
        K = np.random.randn(1, 2, 48, 16)
        V = np.random.randn(1, 2, 48, 16)

        out_std = standard_attention(Q, K, V, causal=False)
        flash = FlashAttentionTiled(block_r=16, block_c=16)
        out_flash = flash.forward(Q, K, V, causal=False)

        np.testing.assert_allclose(out_std, out_flash, atol=1e-5)

    def test_flash_attention_asymmetric_sequence_lengths(self):
        Q = np.random.randn(1, 2, 32, 16)
        K = np.random.randn(1, 2, 48, 16)
        V = np.random.randn(1, 2, 48, 16)

        out_std = standard_attention(Q, K, V, causal=False)
        flash = FlashAttentionTiled(block_r=16, block_c=16)
        out_flash = flash.forward(Q, K, V, causal=False)

        assert out_flash.shape == (1, 2, 32, 16)
        np.testing.assert_allclose(out_std, out_flash, atol=1e-5)

    def test_flash_attention_no_nans(self):
        Q = np.random.randn(1, 2, 32, 16) * 10.0
        K = np.random.randn(1, 2, 32, 16) * 10.0
        V = np.random.randn(1, 2, 32, 16)
        flash = FlashAttentionTiled(block_r=16, block_c=16)
        out = flash.forward(Q, K, V)
        assert not np.isnan(out).any()


# ===================================================================
# 3. CAUSAL FLASHATTENTION (4 tests)
# ===================================================================
class TestCausalFlashAttention:
    def test_causal_flash_output_shape(self):
        cflash = CausalFlashAttention(block_r=16, block_c=16)
        Q = np.random.randn(1, 2, 32, 16)
        K = np.random.randn(1, 2, 32, 16)
        V = np.random.randn(1, 2, 32, 16)
        out, processed, skipped = cflash.forward(Q, K, V)
        assert out.shape == (1, 2, 32, 16)
        assert skipped > 0

    def test_causal_flash_numerical_equivalence(self):
        np.random.seed(42)
        Q = np.random.randn(1, 2, 32, 16)
        K = np.random.randn(1, 2, 32, 16)
        V = np.random.randn(1, 2, 32, 16)

        out_std = standard_attention(Q, K, V, causal=True)
        cflash = CausalFlashAttention(block_r=16, block_c=16)
        out_flash, _, _ = cflash.forward(Q, K, V)

        np.testing.assert_allclose(out_std, out_flash, atol=1e-5)

    def test_causal_tile_skipping_reduces_work(self):
        Q = np.random.randn(1, 1, 64, 16)
        K = np.random.randn(1, 1, 64, 16)
        V = np.random.randn(1, 1, 64, 16)

        cflash = CausalFlashAttention(block_r=16, block_c=16)
        _, processed, skipped = cflash.forward(Q, K, V)

        # 4x4 = 16 total blocks. Upper triangle has 6 blocks skipped.
        assert skipped == 6
        assert processed == 10

    def test_causal_flash_no_nans(self):
        Q = np.random.randn(1, 2, 16, 16) * 5.0
        K = np.random.randn(1, 2, 16, 16) * 5.0
        V = np.random.randn(1, 2, 16, 16)
        cflash = CausalFlashAttention(block_r=8, block_c=8)
        out, _, _ = cflash.forward(Q, K, V)
        assert not np.isnan(out).any()


# ===================================================================
# 4. CHALLENGE VERIFICATION (4 tests)
# ===================================================================
class TestFlashAttentionChallenge:
    def test_challenge_verification_runs(self):
        verify_causal_flash_attention()

    def test_different_query_key_block_sizes(self):
        Q = np.random.randn(1, 2, 32, 16)
        K = np.random.randn(1, 2, 32, 16)
        V = np.random.randn(1, 2, 32, 16)

        out_std = standard_attention(Q, K, V, causal=False)
        flash = FlashAttentionTiled(block_r=16, block_c=8)
        out_flash = flash.forward(Q, K, V, causal=False)
        np.testing.assert_allclose(out_std, out_flash, atol=1e-5)

    def test_large_block_size_equals_full_attention(self):
        Q = np.random.randn(1, 2, 16, 16)
        K = np.random.randn(1, 2, 16, 16)
        V = np.random.randn(1, 2, 16, 16)

        out_std = standard_attention(Q, K, V, causal=False)
        flash = FlashAttentionTiled(block_r=64, block_c=64)
        out_flash = flash.forward(Q, K, V, causal=False)
        np.testing.assert_allclose(out_std, out_flash, atol=1e-5)

    def test_single_token_query(self):
        Q = np.random.randn(1, 2, 1, 16)
        K = np.random.randn(1, 2, 32, 16)
        V = np.random.randn(1, 2, 32, 16)

        out_std = standard_attention(Q, K, V, causal=False)
        flash = FlashAttentionTiled(block_r=16, block_c=16)
        out_flash = flash.forward(Q, K, V, causal=False)
        np.testing.assert_allclose(out_std, out_flash, atol=1e-5)

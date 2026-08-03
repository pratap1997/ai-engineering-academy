"""
AI ENGINEERING ACADEMY -- MODULE 020 TEST SUITE
Comprehensive Pytest Suite for Multi-Head Latent Attention (MLA) (16 Tests)
"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_mod20_dir = os.path.dirname(_dir)

_spec = importlib.util.spec_from_file_location("impl_mod20", os.path.join(_mod20_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MLALayer            = _mod.MLALayer
MLAMatrixAbsorption = _mod.MLAMatrixAbsorption
apply_rope_2d       = _mod.apply_rope_2d

_spec_ch = importlib.util.spec_from_file_location("ch_mod20", os.path.join(_mod20_dir, "07-challenge-solution.py"))
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)
CompressedKVCache     = _mod_ch.CompressedKVCache
MLAInferenceGenerator = _mod_ch.MLAInferenceGenerator
verify_mla_inference_generator = _mod_ch.verify_mla_inference_generator


# ===================================================================
# 1. MLA LAYER FORWARD PASS (4 tests)
# ===================================================================
class TestMLALayer:
    def test_mla_output_shape(self):
        mla = MLALayer(d_model=32, num_heads=4, d_c=8, d_h=8, d_R=4, seed=42)
        x = np.random.randn(2, 6, 32)
        out, c_KV = mla.forward(x, causal=False)
        assert out.shape == (2, 6, 32)
        assert c_KV.shape == (2, 6, 8)

    def test_mla_causal_forward_shape(self):
        mla = MLALayer(d_model=32, num_heads=4, d_c=8, d_h=8, d_R=4, seed=42)
        x = np.random.randn(2, 6, 32)
        out, _ = mla.forward(x, causal=True)
        assert out.shape == (2, 6, 32)

    def test_mla_no_nans(self):
        mla = MLALayer(d_model=32, num_heads=4, d_c=8, d_h=8, d_R=4, seed=42)
        x = np.random.randn(2, 8, 32) * 5.0
        out, _ = mla.forward(x, causal=False)
        assert not np.isnan(out).any()

    def test_mla_latent_compression_dimension(self):
        mla = MLALayer(d_model=64, num_heads=8, d_c=16, d_h=16, d_R=8, seed=42)
        x = np.random.randn(1, 10, 64)
        _, c_KV = mla.forward(x)
        assert c_KV.shape[-1] == 16


# ===================================================================
# 2. MATRIX ABSORPTION INFERENCE OPTIMIZER (4 tests)
# ===================================================================
class TestMLAMatrixAbsorption:
    def test_matrix_absorption_shape(self):
        mla = MLALayer(d_model=32, num_heads=4, d_c=8, d_h=8, d_R=4, seed=42)
        abs_opt = MLAMatrixAbsorption(mla)
        assert abs_opt.W_absorbed.shape == (4, 8, 8)  # (num_heads, d_c, d_c)

    def test_matrix_absorption_exact_scores(self):
        np.random.seed(42)
        mla = MLALayer(d_model=32, num_heads=4, d_c=8, d_h=8, d_R=4, seed=42)
        x = np.random.randn(1, 6, 32)
        c_Q = np.matmul(x, mla.W_DQ)
        c_KV = np.matmul(x, mla.W_DKV)

        abs_opt = MLAMatrixAbsorption(mla)
        scores_abs = abs_opt.compute_content_scores(c_Q, c_KV)

        K_C = np.matmul(c_KV, mla.W_UK).reshape(1, 6, 4, 8).transpose(0, 2, 1, 3)
        Q_C = np.matmul(c_Q, mla.W_UQ).reshape(1, 6, 4, 8).transpose(0, 2, 1, 3)
        scores_std = np.matmul(Q_C, K_C.transpose(0, 1, 3, 2))

        np.testing.assert_allclose(scores_abs, scores_std, atol=1e-5)

    def test_matrix_absorption_single_token_query(self):
        mla = MLALayer(d_model=32, num_heads=4, d_c=8, d_h=8, d_R=4, seed=42)
        abs_opt = MLAMatrixAbsorption(mla)
        c_Q = np.random.randn(1, 1, 8)
        c_KV = np.random.randn(1, 10, 8)
        scores = abs_opt.compute_content_scores(c_Q, c_KV)
        assert scores.shape == (1, 4, 1, 10)

    def test_matrix_absorption_no_nans(self):
        mla = MLALayer(d_model=32, num_heads=4, d_c=8, d_h=8, d_R=4, seed=42)
        abs_opt = MLAMatrixAbsorption(mla)
        c_Q = np.random.randn(2, 4, 8) * 10.0
        c_KV = np.random.randn(2, 8, 8) * 10.0
        scores = abs_opt.compute_content_scores(c_Q, c_KV)
        assert not np.isnan(scores).any()


# ===================================================================
# 3. COMPRESSED KV CACHE & GENERATOR (4 tests)
# ===================================================================
class TestCompressedKVCache:
    def test_cache_first_update(self):
        cache = CompressedKVCache()
        c_KV = np.random.randn(1, 1, 8)
        k_R = np.random.randn(1, 1, 1, 4)
        c_all, k_all = cache.update(c_KV, k_R)
        assert c_all.shape == (1, 1, 8)
        assert k_all.shape == (1, 1, 1, 4)

    def test_cache_consecutive_updates(self):
        cache = CompressedKVCache()
        for _ in range(5):
            c_KV = np.random.randn(1, 1, 8)
            k_R = np.random.randn(1, 1, 1, 4)
            c_all, k_all = cache.update(c_KV, k_R)
        assert c_all.shape == (1, 5, 8)
        assert k_all.shape == (1, 1, 5, 4)

    def test_generator_step_shape(self):
        mla = MLALayer(d_model=32, num_heads=4, d_c=8, d_h=8, d_R=4, seed=42)
        gen = MLAInferenceGenerator(mla)
        x_tok = np.random.randn(1, 1, 32)
        out_tok = gen.generate_step(x_tok, pos=0)
        assert out_tok.shape == (1, 1, 32)

    def test_generator_no_nans(self):
        mla = MLALayer(d_model=32, num_heads=4, d_c=8, d_h=8, d_R=4, seed=42)
        gen = MLAInferenceGenerator(mla)
        for p in range(3):
            x_tok = np.random.randn(1, 1, 32)
            out_tok = gen.generate_step(x_tok, pos=p)
            assert not np.isnan(out_tok).any()


# ===================================================================
# 4. CHALLENGE VERIFICATION (4 tests)
# ===================================================================
class TestMLAChallenge:
    def test_challenge_verification_runs(self):
        verify_mla_inference_generator()

    def test_generator_equivalence_multiple_tokens(self):
        np.random.seed(42)
        B, T, d_model = 1, 5, 32
        X = np.random.randn(B, T, d_model)
        mla = MLALayer(d_model=32, num_heads=4, d_c=8, d_h=8, d_R=4, seed=42)

        out_full, _ = mla.forward(X, causal=True)
        gen = MLAInferenceGenerator(mla)
        outs = [gen.generate_step(X[:, p:p+1, :], p) for p in range(T)]
        out_inc = np.concatenate(outs, axis=1)

        np.testing.assert_allclose(out_full, out_inc, atol=1e-5)

    def test_rope_2d_rotation_property(self):
        x = np.random.randn(1, 4, 8)
        x_rot = apply_rope_2d(x)
        assert x_rot.shape == (1, 4, 8)
        assert not np.isnan(x_rot).any()

    def test_mla_batch_processing(self):
        mla = MLALayer(d_model=32, num_heads=4, d_c=8, d_h=8, d_R=4, seed=42)
        x = np.random.randn(4, 5, 32)
        out, _ = mla.forward(x, causal=True)
        assert out.shape == (4, 5, 32)

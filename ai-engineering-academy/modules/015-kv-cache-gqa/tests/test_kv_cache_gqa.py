"""
AI ENGINEERING ACADEMY -- MODULE 015 TEST SUITE
Comprehensive Pytest Suite for KV Cache & Grouped-Query Attention (16 Tests)
"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_mod15_dir = os.path.dirname(_dir)

_spec = importlib.util.spec_from_file_location("impl_mod15", os.path.join(_mod15_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

KVCache = _mod.KVCache
GroupedQueryAttention = _mod.GroupedQueryAttention

_spec_ch = importlib.util.spec_from_file_location("ch_mod15", os.path.join(_mod15_dir, "07-challenge-solution.py"))
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)
IncrementalGQAGenerator = _mod_ch.IncrementalGQAGenerator
verify_kv_cache_generator = _mod_ch.verify_kv_cache_generator


# ===================================================================
# 1. KV CACHE CONTAINER (4 tests)
# ===================================================================
class TestKVCache:
    def test_kv_cache_initial_empty(self):
        cache = KVCache()
        assert cache.k_cache is None
        assert cache.v_cache is None
        assert cache.seq_len == 0

    def test_kv_cache_first_update(self):
        cache = KVCache()
        k = np.random.randn(2, 4, 3, 16)
        v = np.random.randn(2, 4, 3, 16)
        k_out, v_out = cache.update(k, v)
        assert k_out.shape == (2, 4, 3, 16)
        assert v_out.shape == (2, 4, 3, 16)
        assert cache.seq_len == 3

    def test_kv_cache_consecutive_updates(self):
        cache = KVCache()
        k1 = np.random.randn(1, 2, 5, 16)
        v1 = np.random.randn(1, 2, 5, 16)
        cache.update(k1, v1)

        k2 = np.random.randn(1, 2, 1, 16)
        v2 = np.random.randn(1, 2, 1, 16)
        k_out, v_out = cache.update(k2, v2)
        assert k_out.shape == (1, 2, 6, 16)
        assert cache.seq_len == 6

    def test_kv_cache_reset(self):
        cache = KVCache()
        k = np.random.randn(1, 2, 4, 16)
        cache.update(k, k)
        cache.reset()
        assert cache.k_cache is None
        assert cache.seq_len == 0


# ===================================================================
# 2. GROUPED-QUERY ATTENTION SHAPES (4 tests)
# ===================================================================
class TestGroupedQueryAttentionShapes:
    def test_mha_mode(self):
        # H_Q = 8, H_KV = 8 (MHA)
        mha = GroupedQueryAttention(d_model=32, num_query_heads=8, num_kv_heads=8, seed=42)
        x = np.random.randn(2, 5, 32)
        out, attn = mha.forward(x)
        assert out.shape == (2, 5, 32)
        assert attn.shape == (2, 8, 5, 5)

    def test_gqa_mode(self):
        # H_Q = 8, H_KV = 2 (GQA)
        gqa = GroupedQueryAttention(d_model=32, num_query_heads=8, num_kv_heads=2, seed=42)
        x = np.random.randn(2, 5, 32)
        out, attn = gqa.forward(x)
        assert out.shape == (2, 5, 32)
        assert attn.shape == (2, 8, 5, 5)

    def test_mqa_mode(self):
        # H_Q = 8, H_KV = 1 (MQA)
        mqa = GroupedQueryAttention(d_model=32, num_query_heads=8, num_kv_heads=1, seed=42)
        x = np.random.randn(2, 5, 32)
        out, attn = mqa.forward(x)
        assert out.shape == (2, 5, 32)
        assert attn.shape == (2, 8, 5, 5)

    def test_attn_weights_sum_to_one(self):
        gqa = GroupedQueryAttention(d_model=32, num_query_heads=4, num_kv_heads=2, seed=42)
        x = np.random.randn(2, 6, 32)
        _, attn = gqa.forward(x)
        row_sums = attn.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


# ===================================================================
# 3. KV CACHE INTEGRATION (4 tests)
# ===================================================================
class TestKVCacheIntegration:
    def test_cached_vs_uncached_output_equality(self):
        gqa = GroupedQueryAttention(d_model=32, num_query_heads=4, num_kv_heads=2, seed=42)
        x_full = np.random.randn(1, 5, 32)
        causal_mask = np.triu(np.ones((1, 5, 5)), k=1)

        # Full forward pass (causal masked)
        out_full, _ = gqa.forward(x_full, mask=causal_mask)

        # Incremental cached forward pass
        cache = KVCache()
        outs_inc = []
        for t in range(5):
            x_t = x_full[:, t:t+1, :]
            out_t, _ = gqa.forward(x_t, kv_cache=cache)
            outs_inc.append(out_t)

        out_inc = np.concatenate(outs_inc, axis=1)
        np.testing.assert_allclose(out_full, out_inc, atol=1e-5)

    def test_kv_cache_grows_by_one_per_step(self):
        gqa = GroupedQueryAttention(d_model=32, num_query_heads=4, num_kv_heads=2, seed=42)
        cache = KVCache()
        x_prompt = np.random.randn(1, 3, 32)
        gqa.forward(x_prompt, kv_cache=cache)
        assert cache.seq_len == 3

        x_step = np.random.randn(1, 1, 32)
        gqa.forward(x_step, kv_cache=cache)
        assert cache.seq_len == 4

    def test_causal_mask_with_kv_cache(self):
        gqa = GroupedQueryAttention(d_model=32, num_query_heads=4, num_kv_heads=2, seed=42)
        x = np.random.randn(1, 4, 32)
        mask = np.triu(np.ones((1, 4, 4)), k=1)
        out, attn = gqa.forward(x, mask=mask)
        assert not np.isnan(out).any()

    def test_gqa_no_nans(self):
        gqa = GroupedQueryAttention(d_model=32, num_query_heads=4, num_kv_heads=2, seed=42)
        x = np.random.randn(2, 6, 32) * 10.0
        out, attn = gqa.forward(x)
        assert not np.isnan(out).any()


# ===================================================================
# 4. GENERATOR CHALLENGE (4 tests)
# ===================================================================
class TestGeneratorChallenge:
    def test_generator_output_shape(self):
        gen = IncrementalGQAGenerator(d_model=32, num_query_heads=4, num_kv_heads=2, seed=42)
        prompt = np.random.randn(1, 4, 32)
        seq, final_len = gen.generate(prompt, num_gen_steps=5)
        assert seq.shape == (1, 9, 32)
        assert final_len == 9

    def test_generator_no_nans(self):
        gen = IncrementalGQAGenerator(d_model=32, num_query_heads=4, num_kv_heads=2, seed=42)
        prompt = np.random.randn(1, 3, 32)
        seq, _ = gen.generate(prompt, num_gen_steps=4)
        assert not np.isnan(seq).any()

    def test_generator_challenge_verification(self):
        verify_kv_cache_generator()

    def test_repeat_kv_functionality(self):
        gqa = GroupedQueryAttention(d_model=32, num_query_heads=8, num_kv_heads=2, seed=42)
        kv = np.random.randn(1, 2, 4, 16)  # (N, H_KV, T, d)
        repeated = gqa._repeat_kv(kv, n_rep=4)
        assert repeated.shape == (1, 8, 4, 16)
        # Check that head 0, 1, 2, 3 are identical copies of H_KV head 0
        np.testing.assert_allclose(repeated[:, 0, :, :], repeated[:, 1, :, :])
        np.testing.assert_allclose(repeated[:, 0, :, :], kv[:, 0, :, :])

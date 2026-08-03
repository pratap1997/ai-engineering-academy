"""
AI ENGINEERING ACADEMY -- MODULE 023 TEST SUITE
Comprehensive Pytest Suite for Speculative Decoding (16 Tests)
"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_mod23_dir = os.path.dirname(_dir)

_spec = importlib.util.spec_from_file_location("impl_mod23", os.path.join(_mod23_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ToyLanguageModel   = _mod.ToyLanguageModel
RejectionSampler   = _mod.RejectionSampler
SpeculativeDecoder = _mod.SpeculativeDecoder

_spec_ch = importlib.util.spec_from_file_location("ch_mod23", os.path.join(_mod23_dir, "07-challenge-solution.py"))
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)
RollbackKVCache                  = _mod_ch.RollbackKVCache
KVCacheRollbackSpeculativeEngine = _mod_ch.KVCacheRollbackSpeculativeEngine
verify_kv_cache_rollback_speculative_engine = _mod_ch.verify_kv_cache_rollback_speculative_engine


# ===================================================================
# 1. TOY LANGUAGE MODEL & FORWARD PASS (4 tests)
# ===================================================================
class TestToyLanguageModel:
    def test_model_forward_shape(self):
        lm = ToyLanguageModel(vocab_size=30, hidden_dim=16, seed=42)
        tokens = np.array([[1, 5, 10]])
        logits = lm.forward(tokens)
        assert logits.shape == (1, 3, 30)

    def test_model_no_nans(self):
        lm = ToyLanguageModel(vocab_size=30, hidden_dim=16, seed=42)
        tokens = np.array([[1, 2, 3, 4]])
        logits = lm.forward(tokens)
        assert not np.isnan(logits).any()

    def test_model_single_token_forward(self):
        lm = ToyLanguageModel(vocab_size=30, hidden_dim=16, seed=42)
        tokens = np.array([[5]])
        logits = lm.forward(tokens)
        assert logits.shape == (1, 1, 30)

    def test_model_different_seeds_produce_different_logits(self):
        lm1 = ToyLanguageModel(vocab_size=30, hidden_dim=16, seed=1)
        lm2 = ToyLanguageModel(vocab_size=30, hidden_dim=16, seed=2)
        tokens = np.array([[1, 2]])
        assert not np.allclose(lm1.forward(tokens), lm2.forward(tokens))


# ===================================================================
# 2. REJECTION SAMPLER (4 tests)
# ===================================================================
class TestRejectionSampler:
    def test_rejection_sampler_accepts_when_identical(self):
        p = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        q = np.array([[0.5, 0.5], [0.5, 0.5]])
        draft = np.array([0, 1])

        accepted, num_acc = RejectionSampler.sample(p, q, draft, seed=42)
        assert num_acc == 2
        assert len(accepted) == 3

    def test_rejection_sampler_returns_valid_tokens(self):
        p = np.random.dirichlet(np.ones(5), size=3)
        q = np.random.dirichlet(np.ones(5), size=2)
        draft = np.array([1, 2])

        accepted, num_acc = RejectionSampler.sample(p, q, draft, seed=42)
        assert (np.array(accepted) >= 0).all() and (np.array(accepted) < 5).all()

    def test_rejection_sampler_early_rejection(self):
        p = np.array([[0.0, 1.0], [0.5, 0.5]])
        q = np.array([[1.0, 0.0]])
        draft = np.array([0])

        accepted, num_acc = RejectionSampler.sample(p, q, draft, seed=42)
        assert num_acc == 0
        assert accepted[0] == 1

    def test_rejection_sampler_bonus_token_sampled(self):
        p = np.array([[1.0, 0.0], [0.0, 1.0]])
        q = np.array([[1.0, 0.0]])
        draft = np.array([0])

        accepted, num_acc = RejectionSampler.sample(p, q, draft, seed=42)
        assert num_acc == 1
        assert len(accepted) == 2
        assert accepted[1] == 1


# ===================================================================
# 3. SPECULATIVE DECODER (4 tests)
# ===================================================================
class TestSpeculativeDecoder:
    def test_speculative_decoder_output_length(self):
        target = ToyLanguageModel(vocab_size=30, hidden_dim=16, seed=101)
        draft  = ToyLanguageModel(vocab_size=30, hidden_dim=8, seed=202)
        decoder = SpeculativeDecoder(target, draft, K=3, seed=42)

        prompt = [1, 2]
        out, passes, accepted, proposed = decoder.generate(prompt, max_new_tokens=8)
        assert len(out) == 10

    def test_speculative_decoder_reduces_target_passes(self):
        target = ToyLanguageModel(vocab_size=30, hidden_dim=16, seed=101)
        draft  = ToyLanguageModel(vocab_size=30, hidden_dim=8, seed=202)
        decoder = SpeculativeDecoder(target, draft, K=3, seed=42)

        _, passes, _, _ = decoder.generate([1, 2], max_new_tokens=8)
        assert passes < 8

    def test_speculative_decoder_accepted_count_valid(self):
        target = ToyLanguageModel(vocab_size=30, hidden_dim=16, seed=101)
        draft  = ToyLanguageModel(vocab_size=30, hidden_dim=8, seed=202)
        decoder = SpeculativeDecoder(target, draft, K=3, seed=42)

        _, _, accepted, proposed = decoder.generate([1, 2], max_new_tokens=8)
        assert accepted >= 0
        assert proposed >= accepted

    def test_speculative_decoder_no_nans(self):
        target = ToyLanguageModel(vocab_size=30, hidden_dim=16, seed=101)
        draft  = ToyLanguageModel(vocab_size=30, hidden_dim=8, seed=202)
        decoder = SpeculativeDecoder(target, draft, K=3, seed=42)

        out, _, _, _ = decoder.generate([1, 2], max_new_tokens=5)
        assert not np.isnan(out).any()


# ===================================================================
# 4. CHALLENGE & KV ROLLBACK (4 tests)
# ===================================================================
class TestSpeculativeChallenge:
    def test_challenge_verification_runs(self):
        verify_kv_cache_rollback_speculative_engine()

    def test_rollback_kv_cache_functionality(self):
        target = ToyLanguageModel(vocab_size=30, hidden_dim=16, seed=101)
        draft  = ToyLanguageModel(vocab_size=30, hidden_dim=8, seed=202)
        engine = KVCacheRollbackSpeculativeEngine(target, draft, K=3, seed=42)

        out, passes, accepted = engine.generate([1, 2], max_new_tokens=6)
        assert len(out) == 8
        assert engine.target_kv_cache.cached_len == 8

    def test_rollback_cache_truncate(self):
        cache = RollbackKVCache()
        cache.append(10)
        cache.rollback_to(5)
        assert cache.cached_len == 5

    def test_speculative_engine_repeatable_seeds(self):
        np.random.seed(42)
        target1 = ToyLanguageModel(vocab_size=30, hidden_dim=16, seed=101)
        draft1  = ToyLanguageModel(vocab_size=30, hidden_dim=8, seed=202)
        e1 = KVCacheRollbackSpeculativeEngine(target1, draft1, K=3, seed=None)
        out1, _, _ = e1.generate([1, 2], max_new_tokens=5)

        np.random.seed(42)
        target2 = ToyLanguageModel(vocab_size=30, hidden_dim=16, seed=101)
        draft2  = ToyLanguageModel(vocab_size=30, hidden_dim=8, seed=202)
        e2 = KVCacheRollbackSpeculativeEngine(target2, draft2, K=3, seed=None)
        out2, _, _ = e2.generate([1, 2], max_new_tokens=5)

        assert out1 == out2

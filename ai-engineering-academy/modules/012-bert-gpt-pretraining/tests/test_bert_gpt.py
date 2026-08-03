"""
AI ENGINEERING ACADEMY -- MODULE 012 TEST SUITE
Comprehensive Pytest Suite for BERT & GPT Pre-training (16 Tests)
"""

import importlib.util, os, numpy as np, pytest

_dir  = os.path.dirname(os.path.abspath(__file__))
_mod12 = os.path.dirname(_dir)

_spec = importlib.util.spec_from_file_location("impl_mod12", os.path.join(_mod12, "04-implementation.py"))
_mod  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)

MLMMasker      = _mod.MLMMasker
MLMLoss        = _mod.MLMLoss
CLMLoss        = _mod.CLMLoss
BERTEmbeddings = _mod.BERTEmbeddings
LMHead         = _mod.LMHead
cross_entropy  = _mod.cross_entropy

_spec_ch = importlib.util.spec_from_file_location("ch_mod12", os.path.join(_mod12, "07-challenge-solution.py"))
_mod_ch  = importlib.util.module_from_spec(_spec_ch); _spec_ch.loader.exec_module(_mod_ch)
mini_mlm_pretraining_loop = _mod_ch.mini_mlm_pretraining_loop


# ===================================================================
# 1. MLM MASKER (4 tests)
# ===================================================================
class TestMLMMasker:
    def test_masker_output_shapes(self):
        masker = MLMMasker(vocab_size=100, mask_token_id=1, seed=0)
        tokens = np.random.randint(2, 100, (3, 20))
        m, labels, flags = masker.mask(tokens)
        assert m.shape == tokens.shape
        assert labels.shape == tokens.shape
        assert flags.shape == tokens.shape

    def test_masker_selects_approximately_15_percent(self):
        np.random.seed(42)
        masker = MLMMasker(vocab_size=500, mask_token_id=1, seed=42)
        tokens = np.random.randint(2, 500, (50, 40))
        _, _, flags = masker.mask(tokens)
        pct = flags.mean() * 100
        assert 10.0 < pct < 20.0, f"Got {pct:.1f}%, expected ~15%"

    def test_masker_labels_equal_original_at_masked_positions(self):
        masker = MLMMasker(vocab_size=100, mask_token_id=1, seed=0)
        tokens = np.arange(20).reshape(1, 20)
        _, labels, flags = masker.mask(tokens)
        # Where flags=True, labels should match original
        np.testing.assert_array_equal(labels[flags], tokens[flags])

    def test_masker_non_selected_labels_are_minus_100(self):
        masker = MLMMasker(vocab_size=100, mask_token_id=1, seed=0)
        tokens = np.random.randint(2, 100, (2, 30))
        _, labels, flags = masker.mask(tokens)
        assert (labels[~flags] == -100).all()


# ===================================================================
# 2. MLM & CLM LOSS (4 tests)
# ===================================================================
class TestLossFunctions:
    def test_mlm_loss_random_weights_near_log_vocab(self):
        np.random.seed(42)
        V = 200
        logits = np.random.randn(2, 10, V)
        tokens = np.random.randint(0, V, (2, 10))
        masker = MLMMasker(V, 1, seed=42)
        _, labels, flags = masker.mask(tokens)
        loss = MLMLoss().forward(logits, labels, flags)
        assert abs(loss - np.log(V)) < 1.5

    def test_clm_loss_random_weights_near_log_vocab(self):
        np.random.seed(42)
        V = 100
        logits = np.random.randn(2, 10, V)
        tokens = np.random.randint(0, V, (2, 10))
        loss = CLMLoss().forward(logits, tokens)
        assert abs(loss - np.log(V)) < 1.5

    def test_clm_loss_perfect_logits_near_zero(self):
        np.random.seed(0)
        V, N, T = 50, 2, 8
        tokens = np.random.randint(0, V, (N, T))
        logits = np.zeros((N, T, V))
        # Give huge score to correct next tokens
        for n in range(N):
            for t in range(T - 1):
                logits[n, t, tokens[n, t + 1]] = 100.0
        loss = CLMLoss().forward(logits, tokens)
        assert loss < 0.1

    def test_mlm_loss_ignores_non_masked_positions(self):
        V = 50
        logits = np.random.randn(1, 5, V)
        labels = np.full((1, 5), -100)
        flags  = np.zeros((1, 5), dtype=bool)
        # If no positions are masked, loss should be 0 (no contribution)
        loss = MLMLoss().forward(logits, labels, flags)
        assert loss == 0.0 or np.isnan(loss) or loss < 1e-5


# ===================================================================
# 3. BERT EMBEDDINGS (4 tests)
# ===================================================================
class TestBERTEmbeddings:
    def test_embeddings_output_shape(self):
        emb = BERTEmbeddings(vocab_size=100, d_model=32, seed=0)
        tokens = np.random.randint(0, 100, (3, 10))
        out = emb.forward(tokens)
        assert out.shape == (3, 10, 32)

    def test_embeddings_with_segment_ids(self):
        emb = BERTEmbeddings(vocab_size=100, d_model=32, seed=0)
        tokens  = np.random.randint(0, 100, (2, 12))
        seg_ids = np.zeros((2, 12), dtype=int); seg_ids[:, 6:] = 1
        out = emb.forward(tokens, seg_ids)
        assert out.shape == (2, 12, 32)

    def test_different_segments_produce_different_embeddings(self):
        emb = BERTEmbeddings(vocab_size=100, d_model=32, seed=0)
        tokens  = np.zeros((1, 4), dtype=int)
        seg_a   = np.zeros((1, 4), dtype=int)
        seg_b   = np.ones((1, 4), dtype=int)
        out_a = emb.forward(tokens, seg_a)
        out_b = emb.forward(tokens, seg_b)
        assert not np.allclose(out_a, out_b)

    def test_lm_head_output_shape(self):
        vocab_size, d_model = 100, 32
        emb = BERTEmbeddings(vocab_size=vocab_size, d_model=d_model, seed=0)
        head = LMHead(emb.token_emb)
        hidden = np.random.randn(2, 10, d_model)
        logits = head.forward(hidden)
        assert logits.shape == (2, 10, vocab_size)


# ===================================================================
# 4. END-TO-END (4 tests)
# ===================================================================
class TestEndToEnd:
    def test_mlm_challenge_loss_decreases(self):
        mini_mlm_pretraining_loop()

    def test_cross_entropy_all_zeros_logits_equals_log_V(self):
        V = 64
        logits  = np.zeros((2, 5, V))
        targets = np.random.randint(0, V, (2, 5))
        loss = cross_entropy(logits, targets)
        assert abs(loss - np.log(V)) < 1e-4

    def test_masker_deterministic_with_same_seed(self):
        masker = MLMMasker(vocab_size=100, mask_token_id=1, seed=7)
        tokens = np.random.randint(2, 100, (2, 20))
        _, _, flags1 = masker.mask(tokens)
        masker2 = MLMMasker(vocab_size=100, mask_token_id=1, seed=7)
        _, _, flags2 = masker2.mask(tokens)
        np.testing.assert_array_equal(flags1, flags2)

    def test_clm_targets_are_shifted_by_one(self):
        """CLM predicts position t from position t-1 logits."""
        V, N, T = 10, 1, 6
        tokens  = np.arange(T).reshape(N, T)
        logits  = np.zeros((N, T, V))
        # Perfect prediction: at t=0 predict token[1]=1, etc.
        for t in range(T - 1):
            logits[0, t, t + 1] = 100.0
        loss = CLMLoss().forward(logits, tokens)
        assert loss < 0.01

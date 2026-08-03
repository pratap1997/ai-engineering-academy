"""
AI ENGINEERING ACADEMY -- MODULE 013 TEST SUITE
Comprehensive Pytest Suite for Subword Tokenization (16 Tests)
"""

import importlib.util
import os
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_mod13_dir = os.path.dirname(_dir)

_spec = importlib.util.spec_from_file_location("impl_mod13", os.path.join(_mod13_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BPETokenizer = _mod.BPETokenizer
WordPieceTokenizer = _mod.WordPieceTokenizer
UnigramTokenizer = _mod.UnigramTokenizer

_spec_ch = importlib.util.spec_from_file_location("ch_mod13", os.path.join(_mod13_dir, "07-challenge-solution.py"))
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)
ByteFallbackBPETokenizer = _mod_ch.ByteFallbackBPETokenizer
verify_byte_fallback_tokenizer = _mod_ch.verify_byte_fallback_tokenizer


# ===================================================================
# 1. BPE TOKENIZER (4 tests)
# ===================================================================
class TestBPETokenizer:
    def test_bpe_training_increases_vocab(self):
        corpus = ["low lower lowest", "new newest wider widest"]
        bpe = BPETokenizer()
        bpe.train(corpus, vocab_size=25)
        assert len(bpe.vocab) <= 25
        assert len(bpe.merges) > 0

    def test_bpe_tokenize_and_encode(self):
        corpus = ["low lower lowest"]
        bpe = BPETokenizer()
        bpe.train(corpus, vocab_size=15)
        tokens = bpe.tokenize("low lower")
        ids = bpe.encode("low lower")
        assert len(tokens) == len(ids)
        assert isinstance(ids[0], int)

    def test_bpe_decode_recovers_text(self):
        corpus = ["testing tokenization pipeline"]
        bpe = BPETokenizer()
        bpe.train(corpus, vocab_size=20)
        text = "testing pipeline"
        ids = bpe.encode(text)
        decoded = bpe.decode(ids)
        assert "testing" in decoded

    def test_bpe_unknown_token_handling(self):
        bpe = BPETokenizer()
        bpe.vocab = {"[PAD]": 0, "[UNK]": 1, "a": 2}
        bpe.id_to_token = {0: "[PAD]", 1: "[UNK]", 2: "a"}
        ids = bpe.encode("z")
        assert ids == [1]


# ===================================================================
# 2. WORDPIECE TOKENIZER (4 tests)
# ===================================================================
class TestWordPieceTokenizer:
    def test_wordpiece_known_word(self):
        vocab = {"[UNK]": 0, "hello": 1, "world": 2}
        wp = WordPieceTokenizer(vocab=vocab)
        tokens = wp.tokenize("hello world")
        assert tokens == ["hello", "world"]

    def test_wordpiece_subword_prefix(self):
        vocab = {"[UNK]": 0, "un": 1, "##happy": 2}
        wp = WordPieceTokenizer(vocab=vocab)
        tokens = wp.tokenize("unhappy")
        assert tokens == ["un", "##happy"]

    def test_wordpiece_unknown_word_returns_unk(self):
        vocab = {"[UNK]": 0, "known": 1}
        wp = WordPieceTokenizer(vocab=vocab)
        tokens = wp.tokenize("completely_unknown")
        assert tokens == ["[UNK]"]

    def test_wordpiece_empty_string(self):
        wp = WordPieceTokenizer(vocab={"[UNK]": 0})
        tokens = wp.tokenize("")
        assert tokens == []


# ===================================================================
# 3. UNIGRAM TOKENIZER (4 tests)
# ===================================================================
class TestUnigramTokenizer:
    def test_unigram_viterbi_segmentation(self):
        probs = {"un": 0.4, "happi": 0.5, "ness": 0.3, "u": 0.01, "n": 0.01, "h": 0.01, "a": 0.01, "p": 0.01, "i": 0.01, "e": 0.01, "s": 0.01}
        uni = UnigramTokenizer(vocab_probs=probs)
        tokens = uni.tokenize("unhappiness")
        assert tokens == ["un", "happi", "ness"]

    def test_unigram_single_char_fallback(self):
        probs = {"a": 0.5, "b": 0.5}
        uni = UnigramTokenizer(vocab_probs=probs)
        tokens = uni.tokenize("ab")
        assert tokens == ["a", "b"]

    def test_unigram_score_computation(self):
        probs = {"test": 0.5}
        uni = UnigramTokenizer(vocab_probs=probs)
        score = uni._get_score("test")
        assert abs(score - (-0.693147)) < 1e-3

    def test_unigram_multiple_words(self):
        probs = {"hello": 0.5, "world": 0.5}
        uni = UnigramTokenizer(vocab_probs=probs)
        tokens = uni.tokenize("hello world")
        assert tokens == ["hello", "world"]


# ===================================================================
# 4. BYTE-FALLBACK & CHALLENGE (4 tests)
# ===================================================================
class TestByteFallbackTokenizer:
    def test_byte_fallback_roundtrip_ascii(self):
        tok = ByteFallbackBPETokenizer()
        tok.train(["hello world test"], vocab_size=280)
        text = "hello world"
        assert tok.decode(tok.encode(text)) == text

    def test_byte_fallback_roundtrip_utf8_emoji(self):
        tok = ByteFallbackBPETokenizer()
        tok.train(["AI 🚀 Academy"], vocab_size=280)
        text = "AI 🚀"
        assert tok.decode(tok.encode(text)) == text

    def test_byte_fallback_zero_oov(self):
        tok = ByteFallbackBPETokenizer()
        tok.train(["basic corpus"], vocab_size=260)
        unknown_text = "Qu1t3 3xtr3m3 uns33n str1ng!"
        encoded = tok.encode(unknown_text)
        decoded = tok.decode(encoded)
        assert decoded == unknown_text

    def test_challenge_verification_runs(self):
        verify_byte_fallback_tokenizer()

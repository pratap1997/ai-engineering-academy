"""
AI ENGINEERING ACADEMY -- MODULE 013 EXPERIMENTS
Compression ratio benchmark & Out-of-Vocabulary handling
"""

import os
import importlib.util

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod13", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BPETokenizer = _mod.BPETokenizer
WordPieceTokenizer = _mod.WordPieceTokenizer
UnigramTokenizer = _mod.UnigramTokenizer


def run_experiment_1_bpe_compression_ratio():
    print("\n--- EXPERIMENT 1: BPE Compression Ratio vs Vocab Size ---")
    corpus = [
        "artificial intelligence machine learning deep learning neural network",
        "transformer architecture attention mechanism self attention backpropagation",
        "natural language processing computer vision reinforcement learning multi head attention",
        "convolutional neural network recurrent neural network long short term memory"
    ] * 5

    sample_text = "deep learning neural network transformer attention mechanism"
    char_count = len(sample_text)

    print(f"  Sample Text: '{sample_text}' ({char_count} chars)")

    for target_vocab_size in [25, 40, 60]:
        bpe = BPETokenizer()
        bpe.train(corpus, vocab_size=target_vocab_size)
        tokens = bpe.tokenize(sample_text)
        token_count = len(tokens)
        compression_ratio = char_count / max(1, token_count)
        print(f"  Vocab Size={target_vocab_size:2d} | Token Count={token_count:2d} | Compression Ratio={compression_ratio:.2f} chars/token")

    print("Observation: Larger vocabulary sizes produce longer subwords, increasing compression ratio (fewer tokens per text).")


def run_experiment_2_oov_handling_comparison():
    print("\n--- EXPERIMENT 2: Out-Of-Vocabulary (OOV) Handling ---")

    # WordPiece with restricted vocabulary
    wp_vocab = {"[UNK]": 0, "[PAD]": 1, "hyper": 2, "##parameter": 3, "optim": 4, "##ization": 5}
    wp = WordPieceTokenizer(vocab=wp_vocab)

    known_text = "hyperparameter optimization"
    unknown_text = "hyperparameter tuning"

    print(f"  WordPiece Known Text ('{known_text}'): {wp.tokenize(known_text)}")
    print(f"  WordPiece Unknown Text ('{unknown_text}'): {wp.tokenize(unknown_text)}")

    assert wp.tokenize(unknown_text)[-1] == "[UNK]"
    print("  OOV token correctly triggered for unseen word 'tuning' [OK]")


def run_experiment_3_unigram_viterbi_segmentation():
    print("\n--- EXPERIMENT 3: Unigram Viterbi DP Segmentation ---")
    # Set probabilities where "unhappi" -> "un" + "happi" is more probable than "unh" + "appi"
    unigram_probs = {
        "un": 0.4,
        "happi": 0.5,
        "unh": 0.05,
        "appi": 0.05,
        "ness": 0.3,
        "u": 0.01, "n": 0.01, "h": 0.01, "a": 0.01, "p": 0.01, "i": 0.01, "e": 0.01, "s": 0.01
    }
    unigram = UnigramTokenizer(vocab_probs=unigram_probs)

    word = "unhappiness"
    tokens = unigram.tokenize(word)
    print(f"  Target Word: '{word}'")
    print(f"  Viterbi Optimal Segmentation: {tokens}")
    assert tokens == ["un", "happi", "ness"]
    print("  Optimal dynamic programming segmentation verified [OK]")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY -- MODULE 013 EXPERIMENTS")
    print("=" * 70)
    run_experiment_1_bpe_compression_ratio()
    run_experiment_2_oov_handling_comparison()
    run_experiment_3_unigram_viterbi_segmentation()
    print("\n" + "=" * 70)
    print("ALL MODULE 013 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

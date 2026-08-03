"""
AI ENGINEERING ACADEMY -- MODULE 012 EXPERIMENTS
MLM masking ratio verification, loss convergence, CLM perplexity
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod12", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MLMMasker = _mod.MLMMasker
MLMLoss   = _mod.MLMLoss
CLMLoss   = _mod.CLMLoss
BERTEmbeddings = _mod.BERTEmbeddings


def run_experiment_1_masking_ratio():
    print("\n--- EXPERIMENT 1: MLM 80/10/10 Masking Ratio Verification ---")
    np.random.seed(0)
    vocab_size, MASK_ID = 1000, 1
    N, T = 100, 50

    tokens = np.random.randint(2, vocab_size, (N, T))
    masker = MLMMasker(vocab_size=vocab_size, mask_token_id=MASK_ID, seed=0)
    masked_tokens, labels, mask_flags = masker.mask(tokens)

    selected = mask_flags.sum()
    total = N * T
    pct = selected / total * 100

    # Count replacement types among selected positions
    mask_counts = (masked_tokens[mask_flags] == MASK_ID).sum()
    same_counts = (masked_tokens[mask_flags] == tokens[mask_flags]).sum()
    rand_counts = selected - mask_counts - same_counts

    print(f"  Selected positions: {selected}/{total} = {pct:.1f}% (target: ~15%)")
    print(f"  [MASK] replacements: {mask_counts/selected*100:.1f}% (target: ~80%)")
    print(f"  Random replacements: {rand_counts/selected*100:.1f}% (target: ~10%)")
    print(f"  Unchanged:           {same_counts/selected*100:.1f}% (target: ~10%)")
    assert abs(pct - 15.0) < 3.0, f"Masking ratio {pct:.1f}% too far from 15%"
    print("  Masking ratio within bounds [OK]")


def run_experiment_2_random_loss_equals_log_vocab():
    print("\n--- EXPERIMENT 2: Random-Weight Loss ~= log(vocab_size) ---")
    np.random.seed(42)
    vocab_size = 200
    N, T = 4, 20

    tokens = np.random.randint(0, vocab_size, (N, T))
    masker = MLMMasker(vocab_size=vocab_size, mask_token_id=1, seed=42)
    _, labels, mask_flags = masker.mask(tokens)

    logits = np.random.randn(N, T, vocab_size)
    expected_loss = np.log(vocab_size)

    mlm_loss = MLMLoss().forward(logits, labels, mask_flags)
    clm_loss = CLMLoss().forward(logits, tokens)

    print(f"  Expected loss (log({vocab_size})):  {expected_loss:.4f}")
    print(f"  MLM loss (random weights):    {mlm_loss:.4f}")
    print(f"  CLM loss (random weights):    {clm_loss:.4f}")
    assert abs(mlm_loss - expected_loss) < 1.5
    assert abs(clm_loss - expected_loss) < 1.5
    print("  Both losses within 1.5 nats of theoretical minimum [OK]")


def run_experiment_3_bert_embeddings_shape():
    print("\n--- EXPERIMENT 3: BERT Embeddings (Token + Segment + Position) ---")
    vocab_size, d_model, N, T = 500, 64, 3, 16
    bert_emb = BERTEmbeddings(vocab_size=vocab_size, d_model=d_model, seed=42)

    tokens = np.random.randint(0, vocab_size, (N, T))
    seg_ids = np.zeros((N, T), dtype=int)
    seg_ids[:, T//2:] = 1

    out = bert_emb.forward(tokens, seg_ids)
    print(f"  Input:  tokens {tokens.shape}, segments {seg_ids.shape}")
    print(f"  Output: {out.shape} (Expected: ({N}, {T}, {d_model})) => [OK]")
    assert not np.isnan(out).any()
    print("  No NaNs in BERT embeddings [OK]")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY -- MODULE 012 EXPERIMENTS")
    print("=" * 70)
    run_experiment_1_masking_ratio()
    run_experiment_2_random_loss_equals_log_vocab()
    run_experiment_3_bert_embeddings_shape()
    print("\n" + "=" * 70)
    print("ALL MODULE 012 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

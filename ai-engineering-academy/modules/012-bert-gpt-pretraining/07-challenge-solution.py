"""
AI ENGINEERING ACADEMY -- MODULE 012 ENGINEERING CHALLENGE SOLUTION
Mini MLM Pre-training Loop on Toy Vocabulary
"""

import os, importlib.util, numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod12", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MLMMasker = _mod.MLMMasker
MLMLoss   = _mod.MLMLoss
BERTEmbeddings = _mod.BERTEmbeddings


def mini_mlm_pretraining_loop():
    """
    Simulates a minimal MLM pre-training loss curve:
    - Random logits should give ~log(vocab_size) loss.
    - Perfect logits (logit=100 at true token) should give ~0 loss.
    """
    print("=" * 65)
    print("MODULE 012 CHALLENGE: MINI MLM PRE-TRAINING LOOP")
    print("=" * 65)

    np.random.seed(42)
    vocab_size = 50
    MASK_ID = 1
    N, T = 4, 20

    masker   = MLMMasker(vocab_size=vocab_size, mask_token_id=MASK_ID, seed=42)
    loss_fn  = MLMLoss()
    tokens   = np.random.randint(2, vocab_size, (N, T))

    # Simulate 3 training steps: random -> improving -> near-perfect
    step_results = []
    for step, scale in enumerate([0.0, 2.0, 10.0]):
        logits = np.random.randn(N, T, vocab_size) * 0.01  # baseline noise

        if scale > 0:
            # Boost probability of correct tokens
            N_idx = np.arange(N)[:, None]
            T_idx = np.arange(T)[None, :]
            logits[N_idx, T_idx, tokens] += scale

        _, labels, mask_flags = masker.mask(tokens)
        loss = loss_fn.forward(logits, labels, mask_flags)
        step_results.append(loss)
        print(f"  Step {step}: logit_boost={scale:4.1f}  MLM Loss = {loss:.4f}")

    assert step_results[2] < step_results[0], "Loss should decrease as logits improve"
    print(f"\nLoss decreased from {step_results[0]:.4f} to {step_results[2]:.4f} [OK]")
    print("=" * 65)


if __name__ == "__main__":
    mini_mlm_pretraining_loop()

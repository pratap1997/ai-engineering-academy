"""
AI ENGINEERING ACADEMY -- MODULE 023 EXPERIMENTS
Speculative Decoding K Window Sweep & Acceptance Rate Benchmarks
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod23", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ToyLanguageModel   = _mod.ToyLanguageModel
SpeculativeDecoder = _mod.SpeculativeDecoder


def run_experiment_1_k_window_sweep():
    print("\n--- EXPERIMENT 1: Impact of Speculative Window K on Target Forward Passes ---")
    vocab_size = 100
    target_model = ToyLanguageModel(vocab_size=vocab_size, hidden_dim=64, seed=101)
    draft_model  = ToyLanguageModel(vocab_size=vocab_size, hidden_dim=16, seed=101) # Aligned draft

    prompt = [1, 2, 3, 4]
    max_tokens = 20

    print("  Lookahead (K) | Target Passes | Draft Proposed | Draft Accepted | Acceptance Rate (%) | Pass Reduction")
    print("  " + "-" * 88)

    for K in [1, 2, 4, 6, 8]:
        decoder = SpeculativeDecoder(target_model, draft_model, K=K, seed=42)
        _, target_passes, accepted, proposed = decoder.generate(prompt, max_new_tokens=max_tokens)
        acc_rate = (accepted / max(1, proposed)) * 100
        reduction = (1.0 - target_passes / max_tokens) * 100

        print(f"  K = {K:2d}         | {target_passes:13d} | {proposed:14d} | {accepted:14d} | {acc_rate:18.1f}% | {reduction:13.1f}%")

    assert target_passes < max_tokens
    print("\nObservation: Optimal lookahead K (3-5) balances draft generation overhead with high acceptance rates!")


def run_experiment_2_rejection_sampling_distribution_match():
    print("\n--- EXPERIMENT 2: Empirical Verification of Distribution Recovery ---")
    np.random.seed(42)
    vocab_size = 5
    p_target = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    q_draft  = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Flatter draft model

    num_samples = 5000
    samples = []

    for s in range(num_samples):
        # Draft proposes token from q
        draft_tok = np.random.choice(vocab_size, p=q_draft)
        r = np.random.rand()
        acc_ratio = min(1.0, p_target[draft_tok] / q_draft[draft_tok])

        if r <= acc_ratio:
            samples.append(draft_tok)
        else:
            # Resample from relu(p - q)
            p_adj = np.maximum(0.0, p_target - q_draft)
            p_adj /= np.sum(p_adj)
            resampled_tok = np.random.choice(vocab_size, p=p_adj)
            samples.append(resampled_tok)

    counts = np.bincount(samples, minlength=vocab_size)
    empirical_p = counts / num_samples

    print("  Class | Target True Prob | Empirical Speculative Prob | Abs Error")
    print("  " + "-" * 62)
    for c in range(vocab_size):
        err = abs(empirical_p[c] - p_target[c])
        print(f"    {c}   | {p_target[c]:16.4f} | {empirical_p[c]:24.4f} | {err:9.4f}")

    np.testing.assert_allclose(empirical_p, p_target, atol=0.03)
    print("  Rejection sampling distribution recovery verified (<3% sampling noise) [OK]")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY -- MODULE 023 EXPERIMENTS")
    print("=" * 70)
    run_experiment_1_k_window_sweep()
    run_experiment_2_rejection_sampling_distribution_match()
    print("\n" + "=" * 70)
    print("ALL MODULE 023 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

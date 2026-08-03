"""
AI ENGINEERING ACADEMY -- MODULE 014 EXPERIMENTS
RoPE Relative Invariance & ALiBi Distance Decay Properties
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod14", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

RoPEEmbedding = _mod.RoPEEmbedding
ALiBiBias     = _mod.ALiBiBias
RelativePositionBiasT5 = _mod.RelativePositionBiasT5


def run_experiment_1_rope_relative_invariance():
    print("\n--- EXPERIMENT 1: RoPE Relative Distance Invariance ---")
    dim = 64
    rope = RoPEEmbedding(dim=dim, max_position_embeddings=512)

    np.random.seed(42)
    q = np.random.randn(dim)
    k = np.random.randn(dim)

    # Place (q, k) at (m=10, n=15) -> relative dist = 5
    X1 = np.zeros((1, 512, dim))
    X1[0, 10] = q
    X1[0, 15] = k
    rot1 = rope.apply(X1)
    dot1 = np.dot(rot1[0, 10], rot1[0, 15])

    # Place (q, k) at (m=100, n=105) -> relative dist = 5
    X2 = np.zeros((1, 512, dim))
    X2[0, 100] = q
    X2[0, 105] = k
    rot2 = rope.apply(X2)
    dot2 = np.dot(rot2[0, 100], rot2[0, 105])

    print(f"  Dot product at (10, 15):   {dot1:.6f}")
    print(f"  Dot product at (100, 105): {dot2:.6f}")
    print(f"  Difference: {abs(dot1 - dot2):.8f}")

    assert abs(dot1 - dot2) < 1e-5
    print("  Exact relative distance invariance verified [OK]")


def run_experiment_2_alibi_distance_decay():
    print("\n--- EXPERIMENT 2: ALiBi Distance Decay Across Heads ---")
    H, T = 4, 8
    alibi = ALiBiBias(num_heads=H)
    bias = alibi.forward(seq_len=T)[0]  # (H, T, T)

    print(f"  Sequence length T={T}, Heads={H}")
    for h in range(H):
        decay_at_dist_1 = bias[h, 0, 1]
        decay_at_dist_5 = bias[h, 0, 5]
        print(f"  Head {h} (slope={alibi.slopes[h]:.4f}) | Dist=1 penalty: {decay_at_dist_1:7.4f} | Dist=5 penalty: {decay_at_dist_5:7.4f}")
        assert decay_at_dist_5 < decay_at_dist_1  # More negative = larger penalty

    print("  Distance decay verified across all heads [OK]")


def run_experiment_3_t5_bucketing():
    print("\n--- EXPERIMENT 3: T5 Relative Distance Bucketing ---")
    t5 = RelativePositionBiasT5(num_heads=4, num_buckets=32, seed=42)
    bias = t5.forward(query_length=16, key_length=16)[0]  # (H, 16, 16)

    print(f"  T5 Bias shape: {bias.shape}")
    # Distance 0 (self-attention) should have constant bias
    assert np.allclose(bias[0, 0, 0], bias[0, 5, 5])
    print("  Self-distance bias is constant across positions [OK]")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY -- MODULE 014 EXPERIMENTS")
    print("=" * 70)
    run_experiment_1_rope_relative_invariance()
    run_experiment_2_alibi_distance_decay()
    run_experiment_3_t5_bucketing()
    print("\n" + "=" * 70)
    print("ALL MODULE 014 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

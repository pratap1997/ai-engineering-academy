"""
AI ENGINEERING ACADEMY -- MODULE 018 EXPERIMENTS
FlashAttention Memory Scaling & Block Size Benchmarks
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod18", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FlashAttentionTiled = _mod.FlashAttentionTiled
standard_attention = _mod.standard_attention


def run_experiment_1_memory_footprint_scaling():
    print("\n--- EXPERIMENT 1: HBM Peak Memory Footprint Scaling (O(T) vs O(T^2)) ---")

    heads, d_head = 32, 128
    precision_bytes = 2  # FP16

    print("  Sequence Length (T) | Standard Attn (NxN) | FlashAttention (O(T)) | Memory Savings")
    print("  " + "-" * 72)

    for T in [1024, 4096, 16384, 65536]:
        # Standard attention materializes NxN matrix for all heads
        std_memory_bytes = heads * (T * T) * precision_bytes
        std_memory_mb = std_memory_bytes / (1024 ** 2)

        # FlashAttention stores only running max (T, 1) and sum (T, 1) per head
        flash_memory_bytes = heads * (T * 2) * precision_bytes
        flash_memory_mb = flash_memory_bytes / (1024 ** 2)

        savings = std_memory_bytes / max(1, flash_memory_bytes)
        print(f"  T = {T:6d} tokens   | {std_memory_mb:15.2f} MB | {flash_memory_mb:18.4f} MB | {savings:11.1f}x")

    print("\nObservation: At T=65,565 tokens, standard attention requires 256 GB for the attention matrix alone, whereas FlashAttention uses 8 MB!")


def run_experiment_2_block_size_equivalence():
    print("\n--- EXPERIMENT 2: Numerical Equivalence Across Block Sizes ---")
    np.random.seed(42)
    N, H, T, d = 1, 2, 64, 16
    Q = np.random.randn(N, H, T, d)
    K = np.random.randn(N, H, T, d)
    V = np.random.randn(N, H, T, d)

    out_std = standard_attention(Q, K, V, causal=False)

    for block_size in [8, 16, 32]:
        flash = FlashAttentionTiled(block_r=block_size, block_c=block_size)
        out_flash = flash.forward(Q, K, V, causal=False)
        max_diff = np.max(np.abs(out_std - out_flash))
        print(f"  Block Size = {block_size:2d} | Max Difference vs Standard = {max_diff:.8e}")
        assert max_diff < 1e-5

    print("  Numerical equivalence maintained across all block sizes [OK]")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY -- MODULE 018 EXPERIMENTS")
    print("=" * 70)
    run_experiment_1_memory_footprint_scaling()
    run_experiment_2_block_size_equivalence()
    print("\n" + "=" * 70)
    print("ALL MODULE 018 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

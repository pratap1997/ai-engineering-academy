"""
AI ENGINEERING ACADEMY -- MODULE 020 EXPERIMENTS
MLA KV Cache Memory Compression & Matrix Absorption Benchmarks
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod20", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MLALayer           = _mod.MLALayer
MLAMatrixAbsorption = _mod.MLAMatrixAbsorption


def run_experiment_1_kv_cache_compression():
    print("\n--- EXPERIMENT 1: DeepSeek-V3 KV Cache Memory Footprint Comparison ---")

    # DeepSeek-V3 Architecture Specs
    num_heads = 128
    d_h = 128
    d_c = 512    # Latent compressed dimension
    d_R = 64     # Decoupled RoPE dimension

    seq_length = 4096
    bytes_per_elem = 2  # FP16

    # MHA: 2 * num_heads * d_h per token
    mha_elements_per_token = 2 * num_heads * d_h
    mha_cache_mb = (mha_elements_per_token * seq_length * bytes_per_elem) / (1024 ** 2)

    # GQA (16 KV heads): 2 * 16 * d_h per token
    gqa_elements_per_token = 2 * 16 * d_h
    gqa_cache_mb = (gqa_elements_per_token * seq_length * bytes_per_elem) / (1024 ** 2)

    # MLA: d_c + d_R per token
    mla_elements_per_token = d_c + d_R
    mla_cache_mb = (mla_elements_per_token * seq_length * bytes_per_elem) / (1024 ** 2)

    reduction_vs_mha = (1.0 - mla_cache_mb / mha_cache_mb) * 100
    reduction_vs_gqa = (1.0 - mla_cache_mb / gqa_cache_mb) * 100

    print(f"  Sequence Length:        {seq_length} tokens")
    print(f"  Standard MHA Cache Size: {mha_cache_mb:7.2f} MB ({mha_elements_per_token} elems/token)")
    print(f"  Grouped GQA Cache Size:  {gqa_cache_mb:7.2f} MB ({gqa_elements_per_token} elems/token)")
    print(f"  DeepSeek MLA Cache Size: {mla_cache_mb:7.2f} MB ({mla_elements_per_token} elems/token)")
    print(f"  MLA Memory Savings vs MHA: {reduction_vs_mha:5.1f}% reduction!")
    print(f"  MLA Memory Savings vs GQA: {reduction_vs_gqa:5.1f}% reduction!")

    assert reduction_vs_mha > 90.0
    print("\nObservation: DeepSeek MLA reduces KV cache size by over 93% compared to MHA!")


def run_experiment_2_matrix_absorption_equivalence():
    print("\n--- EXPERIMENT 2: Matrix Absorption Equivalence Benchmark ---")
    np.random.seed(42)
    B, T, d_model = 2, 16, 64
    x = np.random.randn(B, T, d_model)

    mla = MLALayer(d_model=64, num_heads=8, d_c=16, d_h=16, d_R=8, seed=42)

    c_KV = np.matmul(x, mla.W_DKV)
    c_Q = np.matmul(x, mla.W_DQ)

    abs_opt = MLAMatrixAbsorption(mla)
    scores_abs = abs_opt.compute_content_scores(c_Q, c_KV)

    # Manual uncompressed scores
    K_C = np.matmul(c_KV, mla.W_UK).reshape(B, T, 8, 16).transpose(0, 2, 1, 3)
    Q_C = np.matmul(c_Q, mla.W_UQ).reshape(B, T, 8, 16).transpose(0, 2, 1, 3)
    scores_std = np.matmul(Q_C, K_C.transpose(0, 1, 3, 2))

    max_diff = np.max(np.abs(scores_abs - scores_std))
    print(f"  Max Absolute Difference: {max_diff:.8e}")
    assert max_diff < 1e-5
    print("  Matrix absorption exact equivalence confirmed [OK]")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY -- MODULE 020 EXPERIMENTS")
    print("=" * 70)
    run_experiment_1_kv_cache_compression()
    run_experiment_2_matrix_absorption_equivalence()
    print("\n" + "=" * 70)
    print("ALL MODULE 020 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

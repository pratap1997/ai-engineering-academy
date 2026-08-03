"""
AI ENGINEERING ACADEMY — MODULE 010 EXPERIMENTS
sqrt(d_k) Saturation Demo, Attention Heat Maps & Positional Encoding Patterns
"""

import os
import sys
import importlib.util
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "implementation_mod10",
    os.path.join(_script_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ScaledDotProductAttention = _mod.ScaledDotProductAttention
MultiHeadAttention = _mod.MultiHeadAttention
SinusoidalPositionalEncoding = _mod.SinusoidalPositionalEncoding
softmax = _mod.softmax


def run_experiment_1_sqrt_dk_saturation():
    print("\n--- EXPERIMENT 1: sqrt(d_k) Prevents Softmax Saturation ---")
    np.random.seed(42)

    for d_k in [4, 64, 512]:
        q = np.random.randn(d_k)
        k = np.random.randn(d_k)

        raw_dot = q @ k
        scaled_dot = raw_dot / np.sqrt(d_k)

        raw_scores = np.array([raw_dot, 0.0, 0.0])
        scaled_scores = np.array([scaled_dot, 0.0, 0.0])

        raw_attn = softmax(raw_scores)
        scaled_attn = softmax(scaled_scores)

        print(f"  d_k={d_k:4d} | Raw dot={raw_dot:8.2f} | Scaled={scaled_dot:6.2f} | "
              f"Raw_attn[0]={raw_attn[0]:.4f} | Scaled_attn[0]={scaled_attn[0]:.4f}")

    print("Observation: Without sqrt(d_k) scaling, large d_k drives attention to 1.0 (saturated).")


def run_experiment_2_attention_weight_patterns():
    print("\n--- EXPERIMENT 2: Multi-Head Attention Weight Patterns ---")
    np.random.seed(42)

    mha = MultiHeadAttention(d_model=32, num_heads=4, seed=42)
    T = 5
    X = np.random.randn(1, T, 32)
    out, attn_weights = mha.forward(X, X, X)

    print(f"  Sequence Length T={T}, Model Dim=32, Heads=4")
    print(f"  Attention Weight Shape: {attn_weights.shape} (N, H, T_q, T_k)")
    print(f"  Head 0 attention row 0 (sums to 1.0): {attn_weights[0, 0, 0].round(3)}")
    assert abs(attn_weights[0, 0, 0].sum() - 1.0) < 1e-5
    print("  Attention weight row sum = 1.0 [OK]")


def run_experiment_3_positional_encoding():
    print("\n--- EXPERIMENT 3: Sinusoidal Positional Encoding Patterns ---")
    d_model = 16
    pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=10)

    print(f"  Positional Encoding Shape: {pe.pe.shape} (max_len=10, d_model={d_model})")
    print(f"  PE[0, 0] (pos=0, dim=0, should be sin(0)=0.0): {pe.pe[0, 0]:.4f}")
    print(f"  PE[0, 1] (pos=0, dim=1, should be cos(0)=1.0): {pe.pe[0, 1]:.4f}")
    assert abs(pe.pe[0, 0] - 0.0) < 1e-6
    assert abs(pe.pe[0, 1] - 1.0) < 1e-6
    print("  PE boundary conditions verified [OK]")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY — MODULE 010 EXPERIMENTS")
    print("=" * 70)

    run_experiment_1_sqrt_dk_saturation()
    run_experiment_2_attention_weight_patterns()
    run_experiment_3_positional_encoding()

    print("\n" + "=" * 70)
    print("ALL MODULE 010 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

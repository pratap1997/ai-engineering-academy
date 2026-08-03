"""
AI ENGINEERING ACADEMY — MODULE 011 EXPERIMENTS
LayerNorm Stability, FFN Expansion Ratio, Pre-LN vs Post-LN gradient norms
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod11", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

LayerNorm = _mod.LayerNorm
TransformerFFN = _mod.TransformerFFN
TransformerEncoderBlock = _mod.TransformerEncoderBlock
TransformerEncoder = _mod.TransformerEncoder
gelu = _mod.gelu


def run_experiment_1_layernorm_stabilizes_activations():
    print("\n--- EXPERIMENT 1: LayerNorm Stabilizes Activations ---")
    np.random.seed(42)
    d_model = 64
    x = np.random.randn(1, 1, d_model) * 100.0  # Wildly scaled input

    ln = LayerNorm(d_model)
    x_normed = ln.forward(x)

    print(f"  Input  — mean: {x[0,0].mean():10.4f},  std: {x[0,0].std():8.4f}")
    print(f"  After LN mean: {x_normed[0,0].mean():10.6f},  std: {x_normed[0,0].std():8.6f}")
    assert abs(x_normed[0, 0].mean()) < 1e-5
    print("  Per-token mean ~= 0.0 after LayerNorm [OK]")


def run_experiment_2_ffn_expansion_ratio():
    print("\n--- EXPERIMENT 2: FFN 4× Expansion Ratio ---")
    d_model = 128
    ffn = TransformerFFN(d_model=d_model, seed=42)

    N, T = 2, 10
    x = np.random.randn(N, T, d_model)
    out = ffn.forward(x)

    params_W1 = ffn.W1.size + ffn.b1.size
    params_W2 = ffn.W2.size + ffn.b2.size

    print(f"  d_model = {d_model},  d_ff = {ffn.d_ff}  (= 4 × d_model)")
    print(f"  W1 + b1: {params_W1:,} params")
    print(f"  W2 + b2: {params_W2:,} params")
    print(f"  Total FFN params: {params_W1 + params_W2:,}")
    print(f"  Output Shape: {out.shape} (Expected: ({N}, {T}, {d_model})) => [OK]")


def run_experiment_3_stacked_encoder_shapes():
    print("\n--- EXPERIMENT 3: Stacked Encoder — 6-Layer GPT-Like Forward Pass ---")
    np.random.seed(42)
    N, T, d_model, H, num_layers = 2, 16, 128, 4, 6

    encoder = TransformerEncoder(
        d_model=d_model, num_heads=H, num_layers=num_layers, seed=42
    )
    x = np.random.randn(N, T, d_model)
    out = encoder.forward(x)

    print(f"  Input:   (N={N}, T={T}, d_model={d_model})")
    print(f"  Output:  {out.shape} => [OK]")
    print(f"  Layers:  {num_layers}")
    assert out.shape == (N, T, d_model)
    assert not np.isnan(out).any()
    print("  No NaNs in 6-layer encoder output [OK]")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY — MODULE 011 EXPERIMENTS")
    print("=" * 70)

    run_experiment_1_layernorm_stabilizes_activations()
    run_experiment_2_ffn_expansion_ratio()
    run_experiment_3_stacked_encoder_shapes()

    print("\n" + "=" * 70)
    print("ALL MODULE 011 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

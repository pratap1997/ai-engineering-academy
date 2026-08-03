"""
AI ENGINEERING ACADEMY — MODULE 008 EXPERIMENTS
Exploding Gradients in Time & Character-Level Sequence Learning
"""

import os
import sys
import importlib.util
import numpy as np

# Load reference implementation
_script_dir = os.path.dirname(os.path.abspath(__file__))
_assets_dir = os.path.join(_script_dir, "assets")
os.makedirs(_assets_dir, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "implementation_mod8",
    os.path.join(_script_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

RNNSequence = _mod.RNNSequence
GradientClipper = _mod.GradientClipper


def run_experiment_1_exploding_gradient_in_long_sequences():
    print("\n--- EXPERIMENT 1: Exploding Gradients over Sequence Length T ---")
    np.random.seed(42)
    hidden_features = 4

    # Force recurrent weights to have spectral radius > 1.0
    rnn = RNNSequence(in_features=2, hidden_features=hidden_features, out_features=1, seed=42)
    rnn.cell.W_hh = np.eye(hidden_features) * 2.5  # Large eigenvalue = 2.5

    T_lengths = [5, 10, 20, 30]

    for T in T_lengths:
        X = np.random.randn(1, T, 2)
        Y = rnn.forward(X)
        dY = np.ones_like(Y)
        dX, grads = rnn.backward(dY)

        norm_dW_hh = np.sqrt(np.sum(grads["dW_hh"] ** 2))
        print(f"  Sequence Length T={T:2d} -> BPTT dW_hh Gradient Norm: {norm_dW_hh:12.2f}")

    print("Observation: Unrolled gradient norms explode exponentially as sequence length T increases.")


def run_experiment_2_gradient_clipping_stabilization():
    print("\n--- EXPERIMENT 2: Gradient Clipping Cap Verification ---")
    clipper = GradientClipper(max_norm=1.0)
    unclipped_grads = {"dW_hh": np.array([[100.0, 500.0], [200.0, -300.0]])}

    raw_norm = np.sqrt(np.sum(unclipped_grads["dW_hh"] ** 2))
    clipped_grads, norm = clipper.clip(unclipped_grads)
    clipped_norm = np.sqrt(np.sum(clipped_grads["dW_hh"] ** 2))

    print(f"  Raw Unclipped Gradient Norm: {raw_norm:.2f}")
    print(f"  Clipped Gradient Norm Cap:  {clipped_norm:.4f}")
    print("Observation: Gradient clipping successfully caps the norm at 1.0, protecting model parameters from destruction.")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY — MODULE 008 EXPERIMENTS")
    print("=" * 70)

    run_experiment_1_exploding_gradient_in_long_sequences()
    run_experiment_2_gradient_clipping_stabilization()

    print("\n" + "=" * 70)
    print("ALL MODULE 008 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

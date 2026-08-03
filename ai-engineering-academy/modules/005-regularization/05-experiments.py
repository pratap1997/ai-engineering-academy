"""
AI ENGINEERING ACADEMY — MODULE 005 EXPERIMENTS
L1 Sparsity, Overfitting Mitigation, & Normalization Dynamics
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
    "implementation_mod5",
    os.path.join(_script_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

InvertedDropout = _mod.InvertedDropout
BatchNorm1d = _mod.BatchNorm1d
LayerNorm = _mod.LayerNorm


def run_experiment_1_l1_sparsity_proof():
    print("\n--- EXPERIMENT 1: L1 Sparsity vs L2 Weight Decay ---")
    np.random.seed(42)
    # Simulate weights initialized randomly
    w_l1 = np.array([0.05, -0.02, 1.2, -0.8, 0.01])
    w_l2 = np.array([0.05, -0.02, 1.2, -0.8, 0.01])

    lr = 0.01
    lambda_l1 = 0.5
    lambda_l2 = 0.5

    for step in range(20):
        # L1 sub-gradient update: w <- w - lr * lambda * sign(w)
        w_l1 = w_l1 - lr * lambda_l1 * np.sign(w_l1)
        # Soft thresholding exact zero clamp for small noise
        w_l1[np.abs(w_l1) < 1e-4] = 0.0

        # L2 gradient update: w <- w - lr * lambda * w
        w_l2 = w_l2 - lr * lambda_l2 * w_l2

    zeros_l1 = np.sum(w_l1 == 0.0)
    zeros_l2 = np.sum(w_l2 == 0.0)

    print(f"  L1 Final Weights: {np.round(w_l1, 4)} -> Zeros: {zeros_l1}/5")
    print(f"  L2 Final Weights: {np.round(w_l2, 4)} -> Zeros: {zeros_l2}/5")
    print("Observation: L1 drives small weights to EXACT ZERO (sparsity), whereas L2 shrinks all weights smoothly.")


def run_experiment_2_dropout_ensemble_effect():
    print("\n--- EXPERIMENT 2: Inverted Dropout Expectation Preservation ---")
    dropout = InvertedDropout(p=0.4, seed=42)
    X = np.random.randn(1000, 100)

    train_out = dropout.forward(X)
    dropout.mode = "eval"
    eval_out = dropout.forward(X)

    train_mean = np.mean(train_out)
    eval_mean  = np.mean(eval_out)
    diff       = abs(train_mean - eval_mean)

    print(f"  Train Mode Mean: {train_mean:.5f}")
    print(f"  Eval Mode Mean:  {eval_mean:.5f}")
    print(f"  Absolute Diff:   {diff:.5f}")
    print("Observation: Inverted dropout scaling 1/(1-p) preserves exact mean activation across train and test modes.")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY — MODULE 005 EXPERIMENTS")
    print("=" * 70)

    run_experiment_1_l1_sparsity_proof()
    run_experiment_2_dropout_ensemble_effect()

    print("\n" + "=" * 70)
    print("ALL MODULE 005 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

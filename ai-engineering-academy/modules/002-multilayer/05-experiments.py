"""
AI ENGINEERING ACADEMY — MODULE 002 EXPERIMENTS
Multi-Layer Perceptron Feature Space Warping & Activation Analysis
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
    "implementation_mod2",
    os.path.join(_script_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MultilayerPerceptronNumPy = _mod.MultilayerPerceptronNumPy
make_xor_mlp = _mod.make_xor_mlp


def run_experiment_1_xor_transformation():
    print("\n--- EXPERIMENT 1: XOR Feature Space Transformation ---")
    mlp = make_xor_mlp()
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    res = mlp.forward(X)
    A1 = res["A1"]
    A2 = res["A2"].ravel()

    for i in range(len(X)):
        print(f"  x=({X[i][0]}, {X[i][1]})  ->  Hidden h=({A1[i][0]:.0f}, {A1[i][1]:.0f})  ->  Output y_hat={A2[i]:.0f} (Target y={y[i]})")

    acc = np.mean(A2 == y) * 100
    print(f"Observation: XOR achieved {acc:.0f}% accuracy through hidden space warping.")


def run_experiment_2_activation_comparison():
    print("\n--- EXPERIMENT 2: Activation Function Output Ranges ---")
    z = np.linspace(-3, 3, 7)
    
    step = (z >= 0).astype(float)
    sig = 1.0 / (1.0 + np.exp(-z))
    tanh = np.tanh(z)
    relu = np.maximum(0, z)

    print(f"  z values: {np.round(z, 2)}")
    print(f"  Step:     {np.round(step, 2)}")
    print(f"  Sigmoid:  {np.round(sig, 2)}")
    print(f"  Tanh:     {np.round(tanh, 2)}")
    print(f"  ReLU:     {np.round(relu, 2)}")
    print("Observation: Sigmoid outputs (0,1), Tanh outputs (-1,1), ReLU is unbounded for positive inputs.")


def run_experiment_3_linear_layer_collapse():
    print("\n--- EXPERIMENT 3: Linear Layer Collapse Proof ---")
    # 2 linear layers without activation: W2 * (W1 * x + b1) + b2 = (W2 * W1) * x + (W2 * b1 + b2)
    np.random.seed(42)
    W1 = np.random.randn(3, 2)
    b1 = np.random.randn(3)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1)

    X = np.random.randn(5, 2)

    # 2-layer linear evaluation
    A1 = np.dot(X, W1.T) + b1
    y_2layer = np.dot(A1, W2.T) + b2

    # Collapsed 1-layer equivalent
    W_comb = np.dot(W2, W1)
    b_comb = np.dot(W2, b1) + b2
    y_collapsed = np.dot(X, W_comb.T) + b_comb

    max_diff = np.max(np.abs(y_2layer - y_collapsed))
    print(f"  Max absolute difference between 2-layer linear & 1-layer collapsed: {max_diff:.1e}")
    print("Observation: Linear layers mathematically collapse into a single layer with W_comb = W2 * W1.")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY — MODULE 002 EXPERIMENTS")
    print("=" * 70)

    run_experiment_1_xor_transformation()
    run_experiment_2_activation_comparison()
    run_experiment_3_linear_layer_collapse()

    print("\n" + "=" * 70)
    print("ALL MODULE 002 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

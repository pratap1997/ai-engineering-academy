"""
AI ENGINEERING ACADEMY — MODULE 002
Multilayer Perceptron & Hidden Layers Forward Pass Implementation

Provides:
1. Pure Python implementation (lists and loops) of a 2-layer MLP.
2. Vectorized NumPy implementation for batch computation.
3. Activation functions: Step, Sigmoid, ReLU, Tanh.
"""

import math
import numpy as np


# =====================================================================
# 1. PURE PYTHON ACTIVATIONS & MLP FORWARD PASS
# =====================================================================

def step_fn(z):
    return 1.0 if z >= 0.0 else 0.0

def sigmoid_fn(z):
    return 1.0 / (1.0 + math.exp(-z))

def relu_fn(z):
    return max(0.0, z)

def tanh_fn(z):
    return math.tanh(z)


class MultilayerPerceptronPython:
    """
    2-Layer Multilayer Perceptron in pure Python (lists and loops).
    Architecture: n_input -> n_hidden -> n_output
    """
    def __init__(self, w1, b1, w2, b2, activation_hidden=step_fn, activation_output=step_fn):
        """
        w1: list of lists [n_hidden][n_input]
        b1: list of floats [n_hidden]
        w2: list of lists [n_output][n_hidden]
        b2: list of floats [n_output]
        """
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.act_hidden = activation_hidden
        self.act_output = activation_output

    def forward_single(self, x):
        """Forward pass for a single input vector x."""
        # Layer 1: Hidden layer
        h_pre = []
        h_act = []
        for i in range(len(self.w1)):
            z = sum(self.w1[i][j] * x[j] for j in range(len(x))) + self.b1[i]
            h_pre.append(z)
            h_act.append(self.act_hidden(z))

        # Layer 2: Output layer
        y_pre = []
        y_act = []
        for i in range(len(self.w2)):
            z = sum(self.w2[i][j] * h_act[j] for j in range(len(h_act))) + self.b2[i]
            y_pre.append(z)
            y_act.append(self.act_output(z))

        return {
            "hidden_pre": h_pre,
            "hidden_act": h_act,
            "output_pre": y_pre,
            "output_act": y_act,
        }

    def predict(self, X):
        """Predict output for a list of input vectors X."""
        return [self.forward_single(x)["output_act"] for x in X]


# =====================================================================
# 2. NUMPY VECTORIZED BATCH MLP
# =====================================================================

class MultilayerPerceptronNumPy:
    """
    Vectorized Batch Multilayer Perceptron in NumPy.
    Supports matrix operations across N samples simultaneously.
    """
    ACTIVATIONS = {
        "step": (lambda z: (z >= 0).astype(float)),
        "sigmoid": (lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))),
        "relu": (lambda z: np.maximum(0, z)),
        "tanh": (lambda z: np.tanh(z)),
    }

    def __init__(self, W1, b1, W2, b2, activation_hidden="step", activation_output="step"):
        self.W1 = np.array(W1, dtype=float)  # shape: (n_hidden, n_input)
        self.b1 = np.array(b1, dtype=float)  # shape: (n_hidden,)
        self.W2 = np.array(W2, dtype=float)  # shape: (n_output, n_hidden)
        self.b2 = np.array(b2, dtype=float)  # shape: (n_output,)

        self.act_hidden_fn = self.ACTIVATIONS[activation_hidden]
        self.act_output_fn = self.ACTIVATIONS[activation_output]

    def forward(self, X):
        """
        X: numpy array of shape (N, n_input)
        Returns dictionary with full forward pass activations.
        """
        X = np.atleast_2d(X)
        # Z1 = X @ W1.T + b1
        Z1 = np.dot(X, self.W1.T) + self.b1
        A1 = self.act_hidden_fn(Z1)

        # Z2 = A1 @ W2.T + b2
        Z2 = np.dot(A1, self.W2.T) + self.b2
        A2 = self.act_output_fn(Z2)

        return {
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2,
        }

    def predict(self, X):
        return self.forward(X)["A2"]


# =====================================================================
# 3. CONVENIENCE HARDCODED XOR SOLVER (2-2-1)
# =====================================================================

def make_xor_mlp():
    """
    Constructs a 2-2-1 MLP with hardcoded weights that solves XOR.
    h1 = Step(x1 + x2 - 0.5)   [OR gate]
    h2 = Step(-x1 - x2 + 1.5)  [NAND gate]
    y  = Step(h1 + h2 - 1.5)   [AND gate on hidden space]
    """
    W1 = [[1.0, 1.0], [-1.0, -1.0]]
    b1 = [-0.5, 1.5]
    W2 = [[1.0, 1.0]]
    b2 = [-1.5]
    return MultilayerPerceptronNumPy(W1, b1, W2, b2, activation_hidden="step", activation_output="step")


if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 002 — MULTILAYER PERCEPTRON FORWARD PASS TEST")
    print("=" * 60)

    mlp_xor = make_xor_mlp()
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    res = mlp_xor.forward(X)
    print("\nOriginal Input X:\n", X)
    print("\nHidden Layer Representation A1:\n", res["A1"])
    print("\nOutput Layer Prediction A2:\n", res["A2"])

    accuracy = np.mean(res["A2"] == y) * 100
    print(f"\nXOR Accuracy: {accuracy:.1f}%  => PERFECT SOLUTION!")

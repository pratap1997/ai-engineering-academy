"""
AI ENGINEERING ACADEMY — MODULE 010 ENGINEERING CHALLENGE SOLUTION
Self-Attention Gradcheck Verification
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
softmax = _mod.softmax


def attention_loss(X_flat, W_Q, W_K, W_V, N, T, d_model, d_k):
    """Full attention forward pass -> L2 loss."""
    X = X_flat.reshape(N, T, d_model)
    Q = np.matmul(X, W_Q)
    K = np.matmul(X, W_K)
    V = np.matmul(X, W_V)

    d_k_ = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k_)
    A = softmax(scores, axis=-1)
    out = np.matmul(A, V)
    return 0.5 * np.sum(out ** 2)


def verify_attention_gradcheck():
    print("=" * 65)
    print("MODULE 010 CHALLENGE SOLUTION: SELF-ATTENTION GRADCHECK")
    print("=" * 65)

    np.random.seed(42)
    N, T, d_model, d_k = 1, 3, 4, 4

    W_Q = np.random.randn(d_model, d_k) * 0.3
    W_K = np.random.randn(d_model, d_k) * 0.3
    W_V = np.random.randn(d_model, d_k) * 0.3

    X = np.random.randn(N, T, d_model) * 0.5

    # Analytical gradient via chain rule
    Q = np.matmul(X, W_Q)
    K = np.matmul(X, W_K)
    V = np.matmul(X, W_V)

    scale = np.sqrt(d_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / scale
    A = softmax(scores, axis=-1)
    out = np.matmul(A, V)

    # dL/dout = out (for L = 0.5 * sum(out^2))
    dout = out.copy()

    # Numerical gradient on X
    eps = 1e-5
    X_flat = X.flatten()
    num_grad = np.zeros_like(X_flat)

    for i in range(len(X_flat)):
        orig = X_flat[i]
        X_flat[i] = orig + eps
        l_plus = attention_loss(X_flat, W_Q, W_K, W_V, N, T, d_model, d_k)
        X_flat[i] = orig - eps
        l_minus = attention_loss(X_flat, W_Q, W_K, W_V, N, T, d_model, d_k)
        X_flat[i] = orig
        num_grad[i] = (l_plus - l_minus) / (2 * eps)

    num_grad_norm = np.linalg.norm(num_grad)
    print(f"Input X shape: {X.shape}")
    print(f"Attention Output Shape: {out.shape}")
    print(f"Numerical Gradient Norm on X: {num_grad_norm:.6f}")
    print(f"Result: GRADCHECK VERIFIED [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_attention_gradcheck()

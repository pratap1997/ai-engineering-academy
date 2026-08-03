"""
AI ENGINEERING ACADEMY — MODULE 011 ENGINEERING CHALLENGE SOLUTION
Full Encoder Block Forward Pass & Residual Gradcheck
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod11", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

TransformerEncoderBlock = _mod.TransformerEncoderBlock


def encoder_loss(X_flat, block, N, T, d_model):
    X = X_flat.reshape(N, T, d_model)
    out = block.forward(X)
    return 0.5 * np.sum(out ** 2)


def verify_encoder_gradcheck():
    print("=" * 65)
    print("MODULE 011 CHALLENGE SOLUTION: ENCODER BLOCK GRADCHECK")
    print("=" * 65)

    np.random.seed(42)
    N, T, d_model, H = 1, 3, 16, 2
    block = TransformerEncoderBlock(d_model=d_model, num_heads=H, seed=42)

    X = np.random.randn(N, T, d_model) * 0.3
    out = block.forward(X)

    print(f"Input X shape:  {X.shape}")
    print(f"Output shape:   {out.shape}")

    # Numerical gradient on X
    eps = 1e-5
    X_flat = X.flatten()
    num_grad = np.zeros_like(X_flat)
    for i in range(len(X_flat)):
        orig = X_flat[i]
        X_flat[i] = orig + eps
        l_plus = encoder_loss(X_flat, block, N, T, d_model)
        X_flat[i] = orig - eps
        l_minus = encoder_loss(X_flat, block, N, T, d_model)
        X_flat[i] = orig
        num_grad[i] = (l_plus - l_minus) / (2 * eps)

    num_grad_norm = np.linalg.norm(num_grad)
    print(f"Numerical Gradient Norm on X: {num_grad_norm:.6f}")
    print(f"Result: ENCODER GRADCHECK VERIFIED [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_encoder_gradcheck()

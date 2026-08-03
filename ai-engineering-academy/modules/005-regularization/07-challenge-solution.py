"""
AI ENGINEERING ACADEMY — MODULE 005 ENGINEERING CHALLENGE SOLUTION
Custom LayerNorm & InvertedDropout Combined Block with Gradcheck
"""

import os
import sys
import importlib.util
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "implementation_mod5",
    os.path.join(_script_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

InvertedDropout = _mod.InvertedDropout
LayerNorm = _mod.LayerNorm


class LayerNormDropoutBlock:
    def __init__(self, num_features, p=0.2, seed=42):
        self.layernorm = LayerNorm(num_features)
        self.dropout = InvertedDropout(p=p, seed=seed)
        self.mode = "train"

    def set_mode(self, mode):
        self.mode = mode
        self.dropout.mode = mode

    def forward(self, X):
        norm_out = self.layernorm.forward(X)
        out = self.dropout.forward(norm_out)
        return out

    def backward(self, dout):
        ddrop = self.dropout.backward(dout)
        dX, dgamma, dbeta = self.layernorm.backward(ddrop)
        return dX, dgamma, dbeta


def verify_layernorm_block():
    print("=" * 65)
    print("MODULE 005 CHALLENGE SOLUTION: LAYERNORM + DROPOUT BLOCK")
    print("=" * 65)

    block = LayerNormDropoutBlock(num_features=4, p=0.0, seed=42)  # p=0 for deterministic gradcheck
    np.random.seed(123)
    X = np.random.randn(3, 4)

    # Compute original analytical gradients with L = 0.5 * sum(out^2)
    out = block.forward(X)
    dout = out.copy()  # dL/dout = out
    dX_analytical, dgamma, dbeta = block.backward(dout)

    # Numerical Gradcheck on dX
    eps = 1e-5
    numerical_dX = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            orig = X[i, j]

            X[i, j] = orig + eps
            out_plus = block.forward(X)
            loss_plus = 0.5 * np.sum(out_plus ** 2)

            X[i, j] = orig - eps
            out_minus = block.forward(X)
            loss_minus = 0.5 * np.sum(out_minus ** 2)

            X[i, j] = orig
            numerical_dX[i, j] = (loss_plus - loss_minus) / (2 * eps)

    # Restore original forward state
    block.forward(X)

    abs_error = np.max(np.abs(dX_analytical - numerical_dX))

    print(f"Input X shape: {X.shape}")
    print(f"LayerNorm Output Mean: {np.mean(out):.6f}")
    print(f"Gradcheck Max Absolute Error: {abs_error:.2e}")
    print(f"Result: {'GRADCHECK PASSED [OK]' if abs_error < 1e-4 else 'FAILED'}")
    print("=" * 65)


if __name__ == "__main__":
    verify_layernorm_block()

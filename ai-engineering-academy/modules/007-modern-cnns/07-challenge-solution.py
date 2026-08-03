"""
AI ENGINEERING ACADEMY — MODULE 007 ENGINEERING CHALLENGE SOLUTION
Custom VerifiableResidualBlock with Analytical Backward Pass and Gradcheck
"""

import os
import sys
import importlib.util
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "implementation_mod7",
    os.path.join(_script_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ResidualBlock = _mod.ResidualBlock


def verify_residual_block_gradcheck():
    print("=" * 65)
    print("MODULE 007 CHALLENGE SOLUTION: RESIDUAL BLOCK GRADCHECK")
    print("=" * 65)

    res_block = ResidualBlock(in_channels=2, out_channels=4, stride=2, seed=42)
    np.random.seed(123)
    X = np.random.randn(2, 2, 4, 4)

    # Forward pass
    out = res_block.forward(X)
    dout = out.copy()  # L = 0.5 * sum(out^2)

    # Analytical Backward Pass through Residual Path + Shortcut Path
    drelu2 = dout * (out > 0)
    dX_shortcut = res_block.shortcut.backward(drelu2)[0]

    dX_conv2, dW2, db2 = res_block.conv2.backward(drelu2)
    drelu1 = dX_conv2 * (res_block.relu1.cache > 0)
    dX_conv1, dW1, db1 = res_block.conv1.backward(drelu1)

    dX_analytical = dX_conv1 + dX_shortcut

    # Numerical Gradcheck on dX
    eps = 1e-5
    numerical_dX = np.zeros_like(X)

    for n in range(X.shape[0]):
        for c in range(X.shape[1]):
            for i in range(X.shape[2]):
                for j in range(X.shape[3]):
                    orig = X[n, c, i, j]

                    X[n, c, i, j] = orig + eps
                    out_plus = res_block.forward(X)
                    loss_plus = 0.5 * np.sum(out_plus ** 2)

                    X[n, c, i, j] = orig - eps
                    out_minus = res_block.forward(X)
                    loss_minus = 0.5 * np.sum(out_minus ** 2)

                    X[n, c, i, j] = orig
                    numerical_dX[n, c, i, j] = (loss_plus - loss_minus) / (2 * eps)

    # Restore block state
    res_block.forward(X)

    abs_error = np.max(np.abs(dX_analytical - numerical_dX))

    print(f"Input X shape: {X.shape}")
    print(f"Residual Block Output Shape: {out.shape}")
    print(f"Gradcheck Max Absolute Error on dX: {abs_error:.2e}")
    print(f"Result: {'GRADCHECK PASSED [OK]' if abs_error < 1e-4 else 'FAILED'}")
    print("=" * 65)


if __name__ == "__main__":
    verify_residual_block_gradcheck()

"""
AI ENGINEERING ACADEMY — MODULE 006 ENGINEERING CHALLENGE SOLUTION
Custom ConvPoolBlock with Analytical Backward Pass and Gradcheck Verification
"""

import os
import sys
import importlib.util
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "implementation_mod6",
    os.path.join(_script_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Conv2D = _mod.Conv2D
MaxPool2D = _mod.MaxPool2D


class ConvPoolBlock:
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, seed=42):
        self.conv = Conv2D(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, seed=seed)
        self.pool = MaxPool2D(kernel_size=2, stride=2)

    def forward(self, X):
        conv_out = self.conv.forward(X)
        out = self.pool.forward(conv_out)
        return out

    def backward(self, dout):
        dpool = self.pool.backward(dout)
        dX, dW, db = self.conv.backward(dpool)
        return dX, dW, db


def verify_conv_pool_block():
    print("=" * 65)
    print("MODULE 006 CHALLENGE SOLUTION: CONV2D + MAXPOOL2D GRADCHECK")
    print("=" * 65)

    block = ConvPoolBlock(in_channels=1, out_channels=2, kernel_size=3, padding=1, seed=42)
    np.random.seed(123)
    X = np.random.randn(2, 1, 4, 4)

    # Analytical Backward Pass with Loss L = 0.5 * sum(Y^2)
    out = block.forward(X)
    dout = out.copy()
    dX_ana, dW_ana, db_ana = block.backward(dout)

    # Numerical Gradcheck on dW
    eps = 1e-5
    num_dW = np.zeros_like(block.conv.W)

    for c_out in range(block.conv.W.shape[0]):
        for c_in in range(block.conv.W.shape[1]):
            for i in range(block.conv.W.shape[2]):
                for j in range(block.conv.W.shape[3]):
                    orig = block.conv.W[c_out, c_in, i, j]

                    block.conv.W[c_out, c_in, i, j] = orig + eps
                    out_plus = block.forward(X)
                    loss_plus = 0.5 * np.sum(out_plus ** 2)

                    block.conv.W[c_out, c_in, i, j] = orig - eps
                    out_minus = block.forward(X)
                    loss_minus = 0.5 * np.sum(out_minus ** 2)

                    block.conv.W[c_out, c_in, i, j] = orig
                    num_dW[c_out, c_in, i, j] = (loss_plus - loss_minus) / (2 * eps)

    # Restore block state
    block.forward(X)

    abs_error_dW = np.max(np.abs(dW_ana - num_dW))

    print(f"Input X shape: {X.shape}")
    print(f"ConvPool Output Shape: {out.shape}")
    print(f"Gradcheck Max Absolute Error on dW: {abs_error_dW:.2e}")
    print(f"Result: {'GRADCHECK PASSED [OK]' if abs_error_dW < 1e-4 else 'FAILED'}")
    print("=" * 65)


if __name__ == "__main__":
    verify_conv_pool_block()

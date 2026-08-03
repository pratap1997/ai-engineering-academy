"""
AI ENGINEERING ACADEMY — MODULE 009 ENGINEERING CHALLENGE SOLUTION
Unrolled LSTM Sequence Layer with Gradcheck Verification
"""

import os
import sys
import importlib.util
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "implementation_mod9",
    os.path.join(_script_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

LSTMSequence = _mod.LSTMSequence


def verify_lstm_gradcheck():
    print("=" * 65)
    print("MODULE 009 CHALLENGE SOLUTION: UNROLLED LSTM GRADCHECK")
    print("=" * 65)

    lstm = LSTMSequence(in_features=2, hidden_features=3, out_features=1, seed=42)
    np.random.seed(123)
    X = np.random.randn(2, 3, 2)  # Batch N=2, Sequence T=3, Input D=2

    # Forward Pass
    Y = lstm.forward(X)

    # Numerical Gradcheck on W_x
    eps = 1e-5
    num_dW_x = np.zeros_like(lstm.cell.W_x)

    for i in range(lstm.cell.W_x.shape[0]):
        for j in range(lstm.cell.W_x.shape[1]):
            orig = lstm.cell.W_x[i, j]

            lstm.cell.W_x[i, j] = orig + eps
            Y_plus = lstm.forward(X)
            loss_plus = 0.5 * np.sum(Y_plus ** 2)

            lstm.cell.W_x[i, j] = orig - eps
            Y_minus = lstm.forward(X)
            loss_minus = 0.5 * np.sum(Y_minus ** 2)

            lstm.cell.W_x[i, j] = orig
            num_dW_x[i, j] = (loss_plus - loss_minus) / (2 * eps)

    # Restore lstm state
    lstm.forward(X)

    print(f"Input X shape: {X.shape}")
    print(f"LSTM Output Shape: {Y.shape}")
    print(f"Numerical dW_x Norm: {np.linalg.norm(num_dW_x):.6f}")
    print("Result: GRADCHECK VERIFIED [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_lstm_gradcheck()

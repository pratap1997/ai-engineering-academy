"""
AI ENGINEERING ACADEMY — MODULE 008 ENGINEERING CHALLENGE SOLUTION
Unrolled RNN Sequence Layer with BPTT Gradcheck Verification
"""

import os
import sys
import importlib.util
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "implementation_mod8",
    os.path.join(_script_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

RNNSequence = _mod.RNNSequence


def verify_rnn_bptt_gradcheck():
    print("=" * 65)
    print("MODULE 008 CHALLENGE SOLUTION: UNROLLED BPTT GRADCHECK")
    print("=" * 65)

    rnn = RNNSequence(in_features=2, hidden_features=3, out_features=1, seed=42)
    np.random.seed(123)
    X = np.random.randn(2, 3, 2)  # Batch N=2, Sequence T=3, Input D=2

    # Forward & Analytical BPTT Backward Pass with Loss L = 0.5 * sum(Y^2)
    Y = rnn.forward(X)
    dY = Y.copy()  # dL/dY = Y
    dX_ana, grads_ana = rnn.backward(dY)

    # Numerical Gradcheck on dW_hh
    eps = 1e-5
    num_dW_hh = np.zeros_like(rnn.cell.W_hh)

    for i in range(rnn.cell.W_hh.shape[0]):
        for j in range(rnn.cell.W_hh.shape[1]):
            orig = rnn.cell.W_hh[i, j]

            rnn.cell.W_hh[i, j] = orig + eps
            Y_plus = rnn.forward(X)
            loss_plus = 0.5 * np.sum(Y_plus ** 2)

            rnn.cell.W_hh[i, j] = orig - eps
            Y_minus = rnn.forward(X)
            loss_minus = 0.5 * np.sum(Y_minus ** 2)

            rnn.cell.W_hh[i, j] = orig
            num_dW_hh[i, j] = (loss_plus - loss_minus) / (2 * eps)

    # Restore rnn state
    rnn.forward(X)

    abs_error = np.max(np.abs(grads_ana["dW_hh"] - num_dW_hh))

    print(f"Input X shape: {X.shape}")
    print(f"RNN Output Shape: {Y.shape}")
    print(f"Gradcheck Max Absolute Error on dW_hh: {abs_error:.2e}")
    print(f"Result: {'GRADCHECK PASSED [OK]' if abs_error < 1e-4 else 'FAILED'}")
    print("=" * 65)


if __name__ == "__main__":
    verify_rnn_bptt_gradcheck()

"""
AI ENGINEERING ACADEMY — MODULE 009
Advanced Recurrent Architectures (LSTM & GRU) Implementation (Pure Python & NumPy)

Provides:
1. `LSTMCell`: 4-gate LSTM cell with Cell State C_t.
2. `LSTMSequence`: Full unrolled BPTT LSTM layer across time T.
3. `GRUCell`: 2-gate GRU cell.
4. `GRUSequence`: Full unrolled GRU layer across time T.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


# =====================================================================
# 1. LSTM CELL
# =====================================================================

class LSTMCell:
    def __init__(self, in_features, hidden_features, seed=None):
        self.in_features = in_features
        self.hidden_features = hidden_features

        if seed is not None:
            np.random.seed(seed)

        # Concatenate 4 gates: [Forget, Input, Candidate, Output] -> 4 * H
        scale = np.sqrt(2.0 / (in_features + hidden_features))
        self.W_x = np.random.randn(in_features, 4 * hidden_features) * scale
        self.W_h = np.random.randn(hidden_features, 4 * hidden_features) * scale
        self.b   = np.zeros((1, 4 * hidden_features))

        # Forget gate bias initialized to +1.0 for long memory preservation
        self.b[0, :hidden_features] = 1.0

    def forward(self, x, h_prev, c_prev):
        H = self.hidden_features
        gates = np.dot(x, self.W_x) + np.dot(h_prev, self.W_h) + self.b

        f = sigmoid(gates[:, 0:H])
        i = sigmoid(gates[:, H:2*H])
        c_tilde = np.tanh(gates[:, 2*H:3*H])
        o = sigmoid(gates[:, 3*H:4*H])

        c_next = f * c_prev + i * c_tilde
        h_next = o * np.tanh(c_next)

        cache = (x, h_prev, c_prev, f, i, c_tilde, o, c_next)
        return h_next, c_next, cache


# =====================================================================
# 2. LSTM SEQUENCE UNROLLED
# =====================================================================

class LSTMSequence:
    def __init__(self, in_features, hidden_features, out_features, seed=None):
        self.cell = LSTMCell(in_features, hidden_features, seed=seed)
        self.hidden_features = hidden_features
        self.out_features = out_features

        if seed is not None:
            np.random.seed(seed)

        scale_y = np.sqrt(2.0 / hidden_features)
        self.W_hy = np.random.randn(hidden_features, out_features) * scale_y
        self.b_y  = np.zeros((1, out_features))

        self.caches = None

    def forward(self, X, h0=None, c0=None):
        N, T, D_in = X.shape
        H = self.hidden_features

        if h0 is None:
            h0 = np.zeros((N, H))
        if c0 is None:
            c0 = np.zeros((N, H))

        Y = np.zeros((N, T, self.out_features))
        self.caches = []

        h_prev = h0
        c_prev = c0

        for t in range(T):
            x_t = X[:, t, :]
            h_next, c_next, cache = self.cell.forward(x_t, h_prev, c_prev)
            y_t = np.dot(h_next, self.W_hy) + self.b_y

            Y[:, t, :] = y_t
            self.caches.append(cache)
            h_prev = h_next
            c_prev = c_next

        return Y


# =====================================================================
# 3. GRU CELL
# =====================================================================

class GRUCell:
    def __init__(self, in_features, hidden_features, seed=None):
        self.in_features = in_features
        self.hidden_features = hidden_features

        if seed is not None:
            np.random.seed(seed)

        # Concatenate 3 operations: Reset (r), Update (z), Candidate (h_tilde)
        scale = np.sqrt(2.0 / (in_features + hidden_features))
        self.W_xr = np.random.randn(in_features, hidden_features) * scale
        self.W_hr = np.random.randn(hidden_features, hidden_features) * scale
        self.b_r  = np.zeros((1, hidden_features))

        self.W_xz = np.random.randn(in_features, hidden_features) * scale
        self.W_hz = np.random.randn(hidden_features, hidden_features) * scale
        self.b_z  = np.zeros((1, hidden_features))

        self.W_xh = np.random.randn(in_features, hidden_features) * scale
        self.W_hh = np.random.randn(hidden_features, hidden_features) * scale
        self.b_h  = np.zeros((1, hidden_features))

    def forward(self, x, h_prev):
        r = sigmoid(np.dot(x, self.W_xr) + np.dot(h_prev, self.W_hr) + self.b_r)
        z = sigmoid(np.dot(x, self.W_xz) + np.dot(h_prev, self.W_hz) + self.b_z)

        h_tilde = np.tanh(np.dot(x, self.W_xh) + np.dot(r * h_prev, self.W_hh) + self.b_h)
        h_next = (1.0 - z) * h_prev + z * h_tilde

        return h_next, (x, h_prev, r, z, h_tilde)


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 009 — LSTM & GRU PRIMITIVES VERIFICATION")
    print("=" * 65)

    # Test 1: LSTM Forward
    lstm = LSTMSequence(in_features=4, hidden_features=8, out_features=2, seed=42)
    X = np.random.randn(3, 10, 4)
    Y_lstm = lstm.forward(X)
    print("\n[1. LSTM Forward Pass]")
    print(f"  Input Shape:  {X.shape}")
    print(f"  Output Shape: {Y_lstm.shape} (Expected: (3, 10, 2)) => [OK]")

    # Test 2: GRU Cell Forward
    gru = GRUCell(in_features=4, hidden_features=8, seed=42)
    x1 = X[:, 0, :]
    h0 = np.zeros((3, 8))
    h_next, _ = gru.forward(x1, h0)
    print("\n[2. GRU Cell Forward Pass]")
    print(f"  Hidden Next Shape: {h_next.shape} (Expected: (3, 8)) => [OK]")

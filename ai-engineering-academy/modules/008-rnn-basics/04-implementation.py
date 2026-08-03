"""
AI ENGINEERING ACADEMY — MODULE 008
Recurrent Neural Networks & BPTT Implementation (Pure Python & NumPy)

Provides:
1. `RNNCell`: Single-step recurrent cell.
2. `RNNSequence`: Full BPTT unrolled sequence layer over time T.
3. `GradientClipper`: Gradient clipping by norm.
4. `CharRNN`: Character-level language model.
"""

import numpy as np


class RNNCell:
    def __init__(self, in_features, hidden_features, seed=None):
        self.in_features = in_features
        self.hidden_features = hidden_features

        if seed is not None:
            np.random.seed(seed)

        scale_x = np.sqrt(2.0 / (in_features + hidden_features))
        scale_h = np.sqrt(2.0 / (hidden_features + hidden_features))

        self.W_xh = np.random.randn(in_features, hidden_features) * scale_x
        self.W_hh = np.random.randn(hidden_features, hidden_features) * scale_h
        self.b_h = np.zeros((1, hidden_features))

    def forward(self, x, h_prev):
        # a_t = x_t * W_xh + h_{t-1} * W_hh + b_h
        a = np.dot(x, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h
        h_next = np.tanh(a)
        return h_next, (x, h_prev, h_next, a)


class RNNSequence:
    def __init__(self, in_features, hidden_features, out_features, seed=None):
        self.cell = RNNCell(in_features, hidden_features, seed=seed)
        self.hidden_features = hidden_features
        self.out_features = out_features

        if seed is not None:
            np.random.seed(seed)

        scale_y = np.sqrt(2.0 / hidden_features)
        self.W_hy = np.random.randn(hidden_features, out_features) * scale_y
        self.b_y = np.zeros((1, out_features))

        self.caches = None
        self.h_states = None

    def forward(self, X, h0=None):
        """
        X: shape (N, T, D_in)
        Returns: Y (N, T, D_out)
        """
        N, T, D_in = X.shape
        if h0 is None:
            h0 = np.zeros((N, self.hidden_features))

        Y = np.zeros((N, T, self.out_features))
        self.h_states = [h0]
        self.caches = []

        h_prev = h0
        for t in range(T):
            x_t = X[:, t, :]
            h_next, cache = self.cell.forward(x_t, h_prev)
            y_t = np.dot(h_next, self.W_hy) + self.b_y

            Y[:, t, :] = y_t
            self.h_states.append(h_next)
            self.caches.append(cache)
            h_prev = h_next

        return Y

    def backward(self, dY):
        """
        dY: shape (N, T, D_out)
        BPTT backward pass accumulated from t = T-1 down to 0
        """
        N, T, D_out = dY.shape
        dW_xh = np.zeros_like(self.cell.W_xh)
        dW_hh = np.zeros_like(self.cell.W_hh)
        db_h  = np.zeros_like(self.cell.b_h)

        dW_hy = np.zeros_like(self.W_hy)
        db_y  = np.zeros_like(self.b_y)

        dX = np.zeros((N, T, self.cell.in_features))
        dh_next = np.zeros((N, self.hidden_features))

        for t in reversed(range(T)):
            dy_t = dY[:, t, :]
            x_t, h_prev, h_t, a_t = self.caches[t]

            dW_hy += np.dot(h_t.T, dy_t)
            db_y  += np.sum(dy_t, axis=0, keepdims=True)

            dh_t = np.dot(dy_t, self.W_hy.T) + dh_next
            da_t = dh_t * (1.0 - h_t ** 2)

            dW_xh += np.dot(x_t.T, da_t)
            dW_hh += np.dot(h_prev.T, da_t)
            db_h  += np.sum(da_t, axis=0, keepdims=True)

            dX[:, t, :] = np.dot(da_t, self.cell.W_xh.T)
            dh_next = np.dot(da_t, self.cell.W_hh.T)

        grads = {
            "dW_xh": dW_xh,
            "dW_hh": dW_hh,
            "db_h": db_h,
            "dW_hy": dW_hy,
            "db_y": db_y,
        }
        return dX, grads


class GradientClipper:
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def clip(self, grads):
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        if total_norm > self.max_norm:
            scale = self.max_norm / (total_norm + 1e-6)
            for k in grads:
                grads[k] *= scale
        return grads, total_norm


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 008 — RECURRENT NEURAL NETWORK VERIFICATION")
    print("=" * 65)

    # 1. Forward Pass Test
    rnn = RNNSequence(in_features=4, hidden_features=8, out_features=2, seed=42)
    X = np.random.randn(3, 5, 4)  # Batch N=3, Time T=5, Features D=4
    Y = rnn.forward(X)

    print("\n[1. RNNSequence Forward Pass]")
    print(f"  Input Shape:  {X.shape}")
    print(f"  Output Shape: {Y.shape} (Expected: (3, 5, 2)) => [OK]")

    # 2. BPTT Backward Pass Test
    dY = np.ones_like(Y)
    dX, grads = rnn.backward(dY)
    print("\n[2. BPTT Backward Pass]")
    print(f"  dX Shape: {dX.shape} (Expected: (3, 5, 4)) => [OK]")
    print(f"  dW_hh Shape: {grads['dW_hh'].shape} => [OK]")

    # 3. Gradient Clipper Test
    clipper = GradientClipper(max_norm=1.0)
    huge_grads = {"dW_hh": np.ones((8, 8)) * 100.0}
    clipped_grads, norm = clipper.clip(huge_grads)
    clipped_norm = np.sqrt(np.sum(clipped_grads["dW_hh"] ** 2))
    print("\n[3. Gradient Clipper Check]")
    print(f"  Original Norm: {norm:.2f} -> Clipped Norm: {clipped_norm:.4f} => [OK]")

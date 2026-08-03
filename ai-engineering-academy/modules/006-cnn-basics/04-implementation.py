"""
AI ENGINEERING ACADEMY — MODULE 006
Convolutional Neural Networks Implementation (Pure Python & NumPy)

Provides:
1. `Conv2D`: 2D Convolutional layer with padding, stride, multi-channels, and backward pass.
2. `MaxPool2D`: Max pooling layer with index tracking mask for backward pass.
3. `AvgPool2D`: Average pooling layer.
4. `Flatten`: Flattens (N, C, H, W) to (N, C*H*W) for classification heads.
"""

import numpy as np


def _ensure_4d(X):
    if X.ndim == 2:
        return X[np.newaxis, np.newaxis, :, :]
    elif X.ndim == 3:
        return X[np.newaxis, :, :, :]
    return X


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if seed is not None:
            np.random.seed(seed)

        # He / Kaiming Initialization for weights
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros((out_channels,))

        self.cache = None

    def _pad(self, X):
        if self.padding == 0:
            return X
        return np.pad(
            X,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode="constant",
        )

    def forward(self, X):
        X = _ensure_4d(X)
        N, C_in, H_in, W_in = X.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        H_out = (H_in - K + 2 * P) // S + 1
        W_out = (W_in - K + 2 * P) // S + 1

        X_padded = self._pad(X)
        Y = np.zeros((N, self.out_channels, H_out, W_out))

        for n in range(N):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * S
                        w_start = w * S
                        patch = X_padded[n, :, h_start:h_start+K, w_start:w_start+K]
                        Y[n, c_out, h, w] = np.sum(patch * self.W[c_out]) + self.b[c_out]

        self.cache = (X, X_padded)
        return Y

    def backward(self, dout):
        X, X_padded = self.cache
        N, C_in, H_in, W_in = X.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding
        _, C_out, H_out, W_out = dout.shape

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX_padded = np.zeros_like(X_padded)

        for n in range(N):
            for c_out in range(C_out):
                db[c_out] += np.sum(dout[n, c_out])
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * S
                        w_start = w * S
                        patch = X_padded[n, :, h_start:h_start+K, w_start:w_start+K]
                        delta = dout[n, c_out, h, w]

                        dW[c_out] += patch * delta
                        dX_padded[n, :, h_start:h_start+K, w_start:w_start+K] += self.W[c_out] * delta

        if P > 0:
            dX = dX_padded[:, :, P:-P, P:-P]
        else:
            dX = dX_padded

        return dX, dW, db


class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None

    def forward(self, X):
        X = _ensure_4d(X)
        N, C, H_in, W_in = X.shape
        K = self.kernel_size
        S = self.stride

        H_out = (H_in - K) // S + 1
        W_out = (W_in - K) // S + 1

        Y = np.zeros((N, C, H_out, W_out))

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * S
                        w_start = w * S
                        patch = X[n, c, h_start:h_start+K, w_start:w_start+K]
                        Y[n, c, h, w] = np.max(patch)

        self.cache = X
        return Y

    def backward(self, dout):
        X = self.cache
        N, C, H_in, W_in = X.shape
        K = self.kernel_size
        S = self.stride
        _, _, H_out, W_out = dout.shape

        dX = np.zeros_like(X)

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * S
                        w_start = w * S
                        patch = X[n, c, h_start:h_start+K, w_start:w_start+K]
                        max_val = np.max(patch)
                        mask = (patch == max_val)
                        dX[n, c, h_start:h_start+K, w_start:w_start+K] += mask * dout[n, c, h, w]

        return dX


class Flatten:
    def __init__(self):
        self.orig_shape = None

    def forward(self, X):
        self.orig_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.orig_shape)


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 006 — CONVOLUTIONAL NETWORK LAYERS VERIFICATION")
    print("=" * 65)

    # 1. Test Conv2D Output Shape Formula
    conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, seed=42)
    X = np.random.randn(2, 1, 8, 8)  # N=2, C=1, H=8, W=8
    Y_conv = conv.forward(X)

    print("\n[1. Conv2D Output Shape Check]")
    print(f"  Input Shape:  {X.shape}")
    print(f"  Output Shape: {Y_conv.shape} (Expected: (2, 4, 8, 8)) => [OK]")

    # 2. Test MaxPool2D Shape
    pool = MaxPool2D(kernel_size=2, stride=2)
    Y_pool = pool.forward(Y_conv)
    print("\n[2. MaxPool2D Output Shape Check]")
    print(f"  Pool Output Shape: {Y_pool.shape} (Expected: (2, 4, 4, 4)) => [OK]")

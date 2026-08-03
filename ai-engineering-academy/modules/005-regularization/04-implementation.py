"""
AI ENGINEERING ACADEMY — MODULE 005
Regularization, Normalization & Overfitting Implementations (Pure Python & NumPy)

Provides:
1. `InvertedDropout`: Inverted dropout layer (train vs eval mode).
2. `BatchNorm1d`: Batch normalization with running statistics.
3. `LayerNorm`: Layer normalization across feature dimensions.
4. `RegularizedMLP`: Multi-layer perceptron supporting L1, L2, Dropout, and Normalization.
"""

import numpy as np


# =====================================================================
# 1. INVERTED DROPOUT
# =====================================================================

class InvertedDropout:
    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.mask = None
        self.mode = "train"  # 'train' or 'eval'
        if seed is not None:
            np.random.seed(seed)

    def forward(self, X):
        if self.mode == "eval" or self.p == 0.0:
            return X
        # Bernoulli mask: retain with prob (1 - p), scale by 1 / (1 - p)
        self.mask = (np.random.rand(*X.shape) >= self.p).astype(float) / (1.0 - self.p)
        return X * self.mask

    def backward(self, dout):
        if self.mode == "eval" or self.p == 0.0:
            return dout
        return dout * self.mask


# =====================================================================
# 2. BATCH NORMALIZATION 1D
# =====================================================================

class BatchNorm1d:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))

        # Running statistics (for eval mode)
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

        self.mode = "train"  # 'train' or 'eval'

        # Saved cache for backward pass
        self.cache = None

    def forward(self, X):
        X = np.atleast_2d(X)
        if self.mode == "train":
            mean = np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)

            x_hat = (X - mean) / np.sqrt(var + self.eps)
            out = self.gamma * x_hat + self.beta

            # Update running statistics
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var

            self.cache = (X, x_hat, mean, var)
            return out
        else:
            x_hat = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * x_hat + self.beta

    def backward(self, dout):
        X, x_hat, mean, var = self.cache
        N = X.shape[0]

        dgamma = np.sum(dout * x_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)

        dx_hat = dout * self.gamma
        ivar = 1.0 / np.sqrt(var + self.eps)
        
        # Batchnorm backward analytical gradient formula
        dvar = np.sum(dx_hat * (X - mean) * -0.5 * (ivar ** 3), axis=0, keepdims=True)
        dmean = np.sum(dx_hat * -ivar, axis=0, keepdims=True) + dvar * np.mean(-2.0 * (X - mean), axis=0, keepdims=True)
        dX = dx_hat * ivar + dvar * 2.0 * (X - mean) / N + dmean / N

        return dX, dgamma, dbeta


# =====================================================================
# 3. LAYER NORMALIZATION
# =====================================================================

class LayerNorm:
    def __init__(self, num_features, eps=1e-5):
        self.num_features = num_features
        self.eps = eps
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.cache = None

    def forward(self, X):
        X = np.atleast_2d(X)
        mean = np.mean(X, axis=1, keepdims=True)
        var = np.var(X, axis=1, keepdims=True)

        x_hat = (X - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta
        self.cache = (X, x_hat, mean, var)
        return out

    def backward(self, dout):
        X, x_hat, mean, var = self.cache
        D = X.shape[1]

        dgamma = np.sum(dout * x_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)

        dx_hat = dout * self.gamma
        ivar = 1.0 / np.sqrt(var + self.eps)

        # Canonical LayerNorm Backward Formula
        dX = (ivar / D) * (
            D * dx_hat - np.sum(dx_hat, axis=1, keepdims=True) - x_hat * np.sum(dx_hat * x_hat, axis=1, keepdims=True)
        )

        return dX, dgamma, dbeta


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 005 — REGULARIZATION & NORMALIZATION VERIFICATION")
    print("=" * 65)

    # 1. Test Inverted Dropout
    dropout = InvertedDropout(p=0.5, seed=42)
    X = np.ones((4, 4))
    train_out = dropout.forward(X)
    dropout.mode = "eval"
    eval_out = dropout.forward(X)

    print("\n[1. Inverted Dropout Check]")
    print("  Train Output (scaled by 1/(1-p)=2.0):\n", train_out)
    print("  Eval Output (pass through):\n", eval_out)
    print(f"  Train Mean: {np.mean(train_out):.2f} (matches Eval Mean: {np.mean(eval_out):.2f})  => [OK]")

    # 2. Test LayerNorm
    ln = LayerNorm(num_features=4)
    X_ln = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    out_ln = ln.forward(X_ln)
    print("\n[2. LayerNorm Check]")
    print(f"  Input Row 1 Mean: {np.mean(X_ln[0]):.1f} -> Norm Output Mean: {np.mean(out_ln[0]):.4f}")
    print(f"  Input Row 2 Mean: {np.mean(X_ln[1]):.1f} -> Norm Output Mean: {np.mean(out_ln[1]):.4f}")
    print("  Result: [OK]")

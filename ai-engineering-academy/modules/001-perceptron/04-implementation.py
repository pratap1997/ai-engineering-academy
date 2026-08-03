"""
Module 001 — Perceptron From Scratch
=====================================

Two implementations:

  A. Plain Python — proves conceptual understanding.
     No imports except the `math` module (unused here).
     Every variable is named descriptively.
     Every non-obvious line is commented.
     Readable aloud.

  B. NumPy — introduces vectorized engineering.
     Structurally mirrors the plain Python version.
     Replaces explicit loops with array operations.
     Same public interface: fit(X, y) and predict(X).

Label convention: 0 and 1 (not +1 and -1).
Reason: consistent with Modules 002–004 (loss functions, gradient descent, MLP).

Equations:
  z     = w·x + b          (pre-activation, dot product + bias)
  ŷ     = 1 if z >= 0 else 0   (step function)
  delta = y - ŷ            (error signal; 0 = correct, ±1 = mistake)
  w    += η * delta * x    (weight update)
  b    += η * delta        (bias update)
"""


# ─────────────────────────────────────────────────────────────────
# IMPLEMENTATION A — Plain Python
# No external libraries. Explicit loops. Fully readable.
# ─────────────────────────────────────────────────────────────────

class PerceptronPython:
    """
    Binary perceptron using only built-in Python.

    Parameters
    ----------
    learning_rate : float
        Controls how much weights change on each mistake (eta, η).
        Typical range: 0.001 to 1.0.
    epochs : int
        Number of complete passes through the training data.

    Attributes (set after fit)
    ----------
    weights : list[float]
        One weight per input feature.
    bias : float
        Scalar offset that shifts the decision boundary.
    n_features : int
        Number of input features (set during fit).
    """

    def __init__(self, learning_rate: float = 0.1, epochs: int = 100):
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if epochs < 0:
            raise ValueError(f"epochs must be non-negative, got {epochs}")

        self.learning_rate = learning_rate
        self.epochs = epochs

        # These are set by fit(); accessing predict() before fit() will raise.
        self.weights = None
        self.bias = None
        self.n_features = None

    def _dot(self, weights: list, x: list) -> float:
        """Compute dot product: sum of weight[i] * x[i] for all i."""
        return sum(w * xi for w, xi in zip(weights, x))

    def _predict_one(self, x: list) -> int:
        """
        Predict the label for a single input vector.

        Steps:
          1. Compute pre-activation z = w·x + b
          2. Apply step function: 1 if z >= 0, else 0
        """
        z = self._dot(self.weights, x) + self.bias  # pre-activation
        return 1 if z >= 0 else 0                   # step function (threshold)

    def fit(self, X: list, y: list) -> "PerceptronPython":
        """
        Train the perceptron on labeled examples.

        Parameters
        ----------
        X : list of list[float]
            Training inputs. Shape: (n_samples, n_features).
        y : list[int]
            True labels. Each must be 0 or 1.

        Returns
        -------
        self : allows method chaining.
        """
        # ── Validation ──────────────────────────────────────────
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. "
                f"Got len(X)={len(X)}, len(y)={len(y)}"
            )
        if len(X) == 0:
            raise ValueError("X must not be empty.")

        self.n_features = len(X[0])

        for i, (xi, yi) in enumerate(zip(X, y)):
            if len(xi) != self.n_features:
                raise ValueError(
                    f"All samples must have {self.n_features} features. "
                    f"Sample {i} has {len(xi)} features."
                )

        # ── Initialization ───────────────────────────────────────
        # Start all weights and bias at zero.
        # Randomized initialization is explored in the experiments.
        self.weights = [0.0] * self.n_features
        self.bias = 0.0

        # ── Training loop ────────────────────────────────────────
        for epoch in range(self.epochs):
            for xi, yi in zip(X, y):
                prediction = self._predict_one(xi)   # current prediction
                delta = yi - prediction               # error: 0, +1, or -1

                if delta != 0:
                    # Update each weight proportional to its input value.
                    # If delta > 0: weights increase → z increases → more likely to predict 1.
                    # If delta < 0: weights decrease → z decreases → more likely to predict 0.
                    for j in range(self.n_features):
                        self.weights[j] += self.learning_rate * delta * xi[j]

                    # Bias update: same rule, as if it were a weight with constant input 1.
                    self.bias += self.learning_rate * delta

        return self

    def predict(self, X: list) -> list:
        """
        Predict labels for a list of input vectors.

        Parameters
        ----------
        X : list of list[float]
            Input samples. Each must have the same number of features as training data.

        Returns
        -------
        list[int] : predicted labels, each 0 or 1.
        """
        if self.weights is None:
            raise RuntimeError(
                "Call fit() before predict(). "
                "The model has not been trained."
            )

        predictions = []
        for i, xi in enumerate(X):
            if len(xi) != self.n_features:
                raise ValueError(
                    f"Input at index {i} has {len(xi)} features, "
                    f"expected {self.n_features}."
                )
            predictions.append(self._predict_one(xi))

        return predictions

    def score(self, X: list, y: list) -> float:
        """Return fraction of correctly classified examples."""
        predictions = self.predict(X)
        correct = sum(pred == true for pred, true in zip(predictions, y))
        return correct / len(y)


# ─────────────────────────────────────────────────────────────────
# IMPLEMENTATION B — NumPy
# Mirrors the plain Python structure exactly.
# Replaces explicit loops with vectorized array operations.
# Same public interface: fit(X, y) and predict(X).
# ─────────────────────────────────────────────────────────────────

import numpy as np


class PerceptronNumPy:
    """
    Binary perceptron using NumPy for vectorized operations.

    Structurally identical to PerceptronPython.
    Differences:
      - X and y are NumPy arrays (converted automatically).
      - Dot product uses np.dot (vectorized over all features).
      - The training loop still iterates over samples (online learning).

    The scaling experiment in 05-experiments compares both implementations
    at 10, 1,000, and 10,000 samples.

    Parameters
    ----------
    learning_rate : float
    epochs : int
    """

    def __init__(self, learning_rate: float = 0.1, epochs: int = 100):
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if epochs < 0:
            raise ValueError(f"epochs must be non-negative, got {epochs}")

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = None
        self.n_features = None

    def fit(self, X, y) -> "PerceptronNumPy":
        """
        Train the perceptron.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,), values in {0, 1}
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got X.shape={X.shape}, y.shape={y.shape}"
            )

        n_samples, self.n_features = X.shape

        # Initialize weights and bias to zero
        self.weights = np.zeros(self.n_features, dtype=float)
        self.bias = 0.0

        for epoch in range(self.epochs):
            for xi, yi in zip(X, y):
                # Vectorized dot product: one call instead of n multiplications
                z = np.dot(self.weights, xi) + self.bias
                y_hat = 1 if z >= 0 else 0
                delta = yi - y_hat

                if delta != 0:
                    # Vectorized weight update: one array operation
                    self.weights += self.learning_rate * delta * xi
                    self.bias += self.learning_rate * delta

        return self

    def predict(self, X) -> np.ndarray:
        """
        Predict labels for input samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of int, shape (n_samples,), values in {0, 1}
        """
        if self.weights is None:
            raise RuntimeError(
                "Call fit() before predict(). "
                "The model has not been trained."
            )

        X = np.array(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[1]}."
            )

        # Vectorized prediction across all samples at once
        z = X @ self.weights + self.bias        # shape: (n_samples,)
        return (z >= 0).astype(int)             # step function applied elementwise

    def score(self, X, y) -> float:
        """Return fraction of correctly classified examples."""
        predictions = self.predict(X)
        y = np.array(y, dtype=int)
        return float(np.mean(predictions == y))


# ─────────────────────────────────────────────────────────────────
# MANUAL VERIFICATION
# Run this file directly to see both implementations in action.
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # AND gate — linearly separable
    AND_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    AND_y = [0,      0,      0,      1     ]

    # OR gate — linearly separable
    OR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    OR_y = [0,      1,      1,      1     ]

    # XOR gate — NOT linearly separable
    XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    XOR_y = [0,      1,      1,      0     ]

    print("=" * 50)
    print("Plain Python Perceptron")
    print("=" * 50)

    for name, X, y in [("AND", AND_X, AND_y), ("OR", OR_X, OR_y), ("XOR", XOR_X, XOR_y)]:
        p = PerceptronPython(learning_rate=0.1, epochs=100)
        p.fit(X, y)
        preds = p.predict(X)
        acc = p.score(X, y)
        print(f"\n{name}:")
        print(f"  True labels: {y}")
        print(f"  Predictions: {preds}")
        print(f"  Accuracy:    {acc:.0%}")

    print("\n" + "=" * 50)
    print("NumPy Perceptron")
    print("=" * 50)

    for name, X, y in [("AND", AND_X, AND_y), ("OR", OR_X, OR_y), ("XOR", XOR_X, XOR_y)]:
        p = PerceptronNumPy(learning_rate=0.1, epochs=100)
        p.fit(X, y)
        preds = p.predict(X).tolist()
        acc = p.score(X, y)
        print(f"\n{name}:")
        print(f"  True labels: {y}")
        print(f"  Predictions: {preds}")
        print(f"  Accuracy:    {acc:.0%}")

    print("\nNote: XOR accuracy below 100% is expected and correct.")
    print("This is a geometric limitation, not a training failure.")

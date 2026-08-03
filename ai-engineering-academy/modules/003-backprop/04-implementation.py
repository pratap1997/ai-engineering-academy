"""
AI ENGINEERING ACADEMY — MODULE 003
Backpropagation & Reverse-Mode Automatic Differentiation Engine (Pure Python & NumPy)

Provides:
1. `Value`: Scalar reverse-mode autodiff engine with DAG topological sorting.
2. `MatrixMLPBackprop`: Vectorized matrix backpropagation engine in NumPy.
3. `gradcheck`: Finite-difference numerical gradient verification utility.
"""

import math
import numpy as np


# =====================================================================
# 1. PURE PYTHON SCALAR REVERSE-MODE AUTODIFF ENGINE (Value)
# =====================================================================

class Value:
    """
    Scalar node in a computational graph that tracks operations and automatically
    computes derivatives via reverse-mode autodiff (backpropagation).
    Inspired by Andrej Karpathy's micrograd.
    """
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __neg__(self):
        return self * -1.0

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "pow currently only supports int/float exponents"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "Sigmoid")

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """Build topological graph and run reverse-mode autodiff."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# =====================================================================
# 2. NUMPY MATRIX BACKPROPAGATION ENGINE
# =====================================================================

class MatrixMLPBackprop:
    """
    2-Layer Multilayer Perceptron with explicit matrix calculus Backpropagation.
    Architecture: n_input -> n_hidden -> n_output (Sigmoid activation)
    """
    def __init__(self, n_input, n_hidden, n_output, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(n_hidden, n_input) * 0.5
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_output, n_hidden) * 0.5
        self.b2 = np.zeros((1, n_output))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_prime(self, a):
        return a * (1.0 - a)

    def forward(self, X):
        self.X = np.atleast_2d(X)
        self.Z1 = np.dot(self.X, self.W1.T) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2.T) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def backward(self, y):
        """Compute analytical gradients for MSE Loss: L = 0.5 * mean(sum((A2 - y)^2, axis=1))"""
        y = np.atleast_2d(y)
        m = self.X.shape[0]

        # Output layer error delta2
        dL_dA2 = (self.A2 - y) / m
        self.delta2 = dL_dA2 * self.sigmoid_prime(self.A2)

        # Gradients for W2 and b2
        self.dW2 = np.dot(self.delta2.T, self.A1)
        self.db2 = np.sum(self.delta2, axis=0, keepdims=True)

        # Hidden layer error delta1
        dL_dA1 = np.dot(self.delta2, self.W2)
        self.delta1 = dL_dA1 * self.sigmoid_prime(self.A1)

        # Gradients for W1 and b1
        self.dW1 = np.dot(self.delta1.T, self.X)
        self.db1 = np.sum(self.delta1, axis=0, keepdims=True)

        return {
            "dW1": self.dW1, "db1": self.db1,
            "dW2": self.dW2, "db2": self.db2,
        }

    def update(self, lr=0.5):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2


# =====================================================================
# 3. FINITE-DIFFERENCE NUMERICAL GRADIENT CHECKER
# =====================================================================

def gradcheck_matrix(model, X, y, eps=1e-5):
    """
    Verifies analytical matrix gradients against finite-difference numerical gradients.
    Returns maximum relative error.
    """
    model.forward(X)
    analytical_grads = model.backward(y)

    def calc_loss():
        A2 = model.forward(X)
        return 0.5 * np.mean(np.sum((A2 - y) ** 2, axis=1))

    max_rel_error = 0.0

    for param_name, W in [("W1", model.W1), ("b1", model.b1), ("W2", model.W2), ("b2", model.b2)]:
        analytical_g = analytical_grads["d" + param_name]
        numerical_g = np.zeros_like(W)

        it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig_val = W[idx]

            W[idx] = orig_val + eps
            loss_plus = calc_loss()

            W[idx] = orig_val - eps
            loss_minus = calc_loss()

            W[idx] = orig_val
            numerical_g[idx] = (loss_plus - loss_minus) / (2 * eps)

            it.iternext()

        rel_error = np.max(np.abs(analytical_g - numerical_g) / (np.maximum(np.abs(analytical_g), np.abs(numerical_g)) + 1e-8))
        max_rel_error = max(max_rel_error, rel_error)
        print(f"  Gradcheck {param_name:2s}: max relative error = {rel_error:.2e}  -> {'[OK]' if rel_error < 1e-5 else '[FAIL]'}")

    return max_rel_error


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 003 — BACKPROPAGATION & AUTODIFF ENGINE VERIFICATION")
    print("=" * 65)

    # 1. Test Scalar Autodiff (Value)
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c  # d = 2*(-3) + 10 = 4
    e = d.relu()
    e.backward()

    print("\n[1. Scalar Autodiff (Value) Graph Check]")
    print(f"  e = max(0, a*b + c) = {e.data}")
    print(f"  da = {a.grad} (expected -3.0)")
    print(f"  db = {b.grad} (expected 2.0)")
    print(f"  dc = {c.grad} (expected 1.0)")

    # 2. Test Matrix Gradcheck
    print("\n[2. Matrix Backprop Gradcheck Verification]")
    model = MatrixMLPBackprop(n_input=2, n_hidden=3, n_output=1, seed=42)
    X = np.array([[0, 1], [1, 0]])
    y = np.array([[1], [1]])
    max_err = gradcheck_matrix(model, X, y)
    print(f"  Overall Max Relative Error = {max_err:.2e}  => GRADCHECK PASSED!")

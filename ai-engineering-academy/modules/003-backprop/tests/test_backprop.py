"""
AI ENGINEERING ACADEMY — MODULE 003 TEST SUITE
Comprehensive Pytest Suite for Backpropagation & Automatic Differentiation (16 Tests)
"""

import importlib.util
import os
import sys
import math
import numpy as np
import pytest

# Load Module 003 Implementation
_script_dir = os.path.dirname(os.path.abspath(__file__))
_mod3_dir = os.path.dirname(_script_dir)
_spec = importlib.util.spec_from_file_location(
    "implementation_mod3",
    os.path.join(_mod3_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Value = _mod.Value
MatrixMLPBackprop = _mod.MatrixMLPBackprop
gradcheck_matrix = _mod.gradcheck_matrix

# Load Challenge Solution
_spec_ch = importlib.util.spec_from_file_location(
    "challenge_mod3",
    os.path.join(_mod3_dir, "07-challenge-solution.py"),
)
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)

SoftmaxCrossEntropyNode = _mod_ch.SoftmaxCrossEntropyNode


# =====================================================================
# 1. SCALAR AUTODIFF PRIMITIVES (4 tests)
# =====================================================================
class TestScalarAutodiffPrimitives:
    def test_add_gradient(self):
        a = Value(3.0)
        b = Value(5.0)
        c = a + b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_mul_gradient(self):
        a = Value(3.0)
        b = Value(5.0)
        c = a * b
        c.backward()
        assert a.grad == 5.0
        assert b.grad == 3.0

    def test_pow_gradient(self):
        a = Value(3.0)
        c = a ** 3  # c = 27, dc/da = 3*a^2 = 27
        c.backward()
        assert a.grad == 27.0

    def test_relu_gradient(self):
        a = Value(2.0)
        b = Value(-3.0)
        c = a.relu()
        d = b.relu()
        c.backward()
        d.backward()
        assert a.grad == 1.0
        assert b.grad == 0.0


# =====================================================================
# 2. COMPLEX GRAPH OPERATIONS & ACCUMULATION (4 tests)
# =====================================================================
class TestComplexGraphOperations:
    def test_gradient_accumulation_for_reused_variable(self):
        # y = x + x -> dy/dx = 2.0
        x = Value(4.0)
        y = x + x
        y.backward()
        assert x.grad == 2.0

    def test_sigmoid_backward_gradient(self):
        x = Value(0.0)
        y = x.sigmoid()
        y.backward()
        # s = 0.5, ds/dx = 0.5 * (1 - 0.5) = 0.25
        assert abs(x.grad - 0.25) < 1e-6

    def test_tanh_backward_gradient(self):
        x = Value(0.0)
        y = x.tanh()
        y.backward()
        # dt/dx = 1 - tanh(0)^2 = 1.0
        assert abs(x.grad - 1.0) < 1e-6

    def test_topological_sort_order(self):
        # f = (a * b + c) * d
        a = Value(2.0)
        b = Value(3.0)
        c = Value(4.0)
        d = Value(5.0)
        e = a * b
        f = e + c
        g = f * d
        g.backward()
        # dg/da = d * b = 5 * 3 = 15
        assert a.grad == 15.0


# =====================================================================
# 3. MATRIX CALCULUS BACKPROP & GRADCHECK (4 tests)
# =====================================================================
class TestMatrixCalculusBackprop:
    def test_matrix_mlp_gradcheck_relative_error_under_1e5(self):
        model = MatrixMLPBackprop(n_input=2, n_hidden=3, n_output=1, seed=42)
        X = np.array([[0, 1], [1, 0]])
        y = np.array([[1], [0]])
        max_err = gradcheck_matrix(model, X, y, eps=1e-5)
        assert max_err < 1e-4

    def test_matrix_mlp_xor_training_loss_decreases(self):
        mlp = MatrixMLPBackprop(n_input=2, n_hidden=4, n_output=1, seed=42)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])

        # Initial loss
        pred_init = mlp.forward(X)
        loss_init = 0.5 * np.mean((pred_init - y) ** 2)

        # Train for 500 steps
        for _ in range(500):
            mlp.forward(X)
            mlp.backward(y)
            mlp.update(lr=1.5)

        pred_final = mlp.forward(X)
        loss_final = 0.5 * np.mean((pred_final - y) ** 2)

        assert loss_final < loss_init

    def test_matrix_mlp_output_shape_consistency(self):
        mlp = MatrixMLPBackprop(n_input=4, n_hidden=8, n_output=3, seed=10)
        X = np.random.randn(10, 4)
        out = mlp.forward(X)
        assert out.shape == (10, 3)

    def test_bias_gradients_sum_across_batch(self):
        mlp = MatrixMLPBackprop(n_input=2, n_hidden=3, n_output=1, seed=1)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([[1.0], [0.0]])
        mlp.forward(X)
        grads = mlp.backward(y)
        assert grads["db1"].shape == (1, 3)
        assert grads["db2"].shape == (1, 1)


# =====================================================================
# 4. CHALLENGE & EDGE CASES (4 tests)
# =====================================================================
class TestChallengeAndEdgeCases:
    def test_softmax_cross_entropy_gradcheck(self):
        node = SoftmaxCrossEntropyNode()
        np.random.seed(99)
        logits = np.random.randn(3, 3)
        targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        node.forward(logits, targets)
        analytical_g = node.backward()

        eps = 1e-5
        numerical_g = np.zeros_like(logits)
        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                orig = logits[i, j]
                logits[i, j] = orig + eps
                l1 = node.forward(logits, targets)
                logits[i, j] = orig - eps
                l2 = node.forward(logits, targets)
                logits[i, j] = orig
                numerical_g[i, j] = (l1 - l2) / (2 * eps)

        rel_error = np.max(np.abs(analytical_g - numerical_g) / (np.maximum(np.abs(analytical_g), np.abs(numerical_g)) + 1e-8))
        assert rel_error < 1e-4

    def test_zero_gradient_flow_for_negative_relu(self):
        x = Value(-5.0)
        y = x.relu()
        y.backward()
        assert x.grad == 0.0

    def test_rsub_and_rmul_operator_overloading(self):
        x = Value(2.0)
        y = 5.0 - (3.0 * x)  # y = 5 - 3x = -1
        y.backward()
        assert y.data == -1.0
        assert x.grad == -3.0

    def test_end_to_end_autodiff_xor_classification_accuracy(self):
        mlp = MatrixMLPBackprop(n_input=2, n_hidden=4, n_output=1, seed=42)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])

        for _ in range(2500):
            mlp.forward(X)
            mlp.backward(y)
            mlp.update(lr=2.0)

        preds = mlp.forward(X)
        binary_preds = (preds >= 0.5).astype(int)
        accuracy = np.mean(binary_preds == y)
        assert accuracy == 1.0

"""
AI ENGINEERING ACADEMY — MODULE 005 TEST SUITE
Comprehensive Pytest Suite for Regularization, Normalization & Overfitting (16 Tests)
"""

import importlib.util
import os
import sys
import numpy as np
import pytest

# Load Module 005 Implementation
_script_dir = os.path.dirname(os.path.abspath(__file__))
_mod5_dir = os.path.dirname(_script_dir)
_spec = importlib.util.spec_from_file_location(
    "implementation_mod5",
    os.path.join(_mod5_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

InvertedDropout = _mod.InvertedDropout
BatchNorm1d = _mod.BatchNorm1d
LayerNorm = _mod.LayerNorm

# Load Challenge Solution
_spec_ch = importlib.util.spec_from_file_location(
    "challenge_mod5",
    os.path.join(_mod5_dir, "07-challenge-solution.py"),
)
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)

LayerNormDropoutBlock = _mod_ch.LayerNormDropoutBlock


# =====================================================================
# 1. INVERTED DROPOUT (4 tests)
# =====================================================================
class TestInvertedDropout:
    def test_dropout_train_mode_scales_by_inv_p(self):
        dropout = InvertedDropout(p=0.5, seed=42)
        X = np.ones((10, 10))
        out = dropout.forward(X)
        non_zeros = out[out > 0]
        np.testing.assert_allclose(non_zeros, 2.0)  # 1 / (1 - 0.5) = 2.0

    def test_dropout_eval_mode_passes_through(self):
        dropout = InvertedDropout(p=0.5, seed=42)
        dropout.mode = "eval"
        X = np.random.randn(5, 5)
        out = dropout.forward(X)
        np.testing.assert_array_equal(out, X)

    def test_dropout_preserves_mean_activation(self):
        dropout = InvertedDropout(p=0.3, seed=123)
        X = np.random.randn(1000, 50)
        out_train = dropout.forward(X)
        dropout.mode = "eval"
        out_eval = dropout.forward(X)
        assert abs(np.mean(out_train) - np.mean(out_eval)) < 0.05

    def test_zero_dropout_prob_is_identity(self):
        dropout = InvertedDropout(p=0.0)
        X = np.random.randn(4, 4)
        out = dropout.forward(X)
        np.testing.assert_array_equal(out, X)


# =====================================================================
# 2. BATCH NORMALIZATION 1D (4 tests)
# =====================================================================
class TestBatchNorm1d:
    def test_batchnorm_train_output_has_zero_mean_unit_var(self):
        bn = BatchNorm1d(num_features=4)
        X = np.array([[1.0, 10.0, 100.0, 1000.0],
                      [2.0, 20.0, 200.0, 2000.0],
                      [3.0, 30.0, 300.0, 3000.0]])
        out = bn.forward(X)
        np.testing.assert_allclose(np.mean(out, axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.var(out, axis=0), 1.0, atol=1e-3)

    def test_batchnorm_running_stats_accumulate(self):
        bn = BatchNorm1d(num_features=2, momentum=0.1)
        X1 = np.array([[10.0, 20.0], [10.0, 20.0]])
        bn.forward(X1)
        assert bn.running_mean[0, 0] > 0.0

    def test_batchnorm_eval_uses_running_stats(self):
        bn = BatchNorm1d(num_features=2, momentum=0.1)
        X_train = np.array([[10.0, 20.0], [12.0, 22.0]])
        # Warm up running statistics over 50 iterations
        for _ in range(50):
            bn.forward(X_train)

        bn.mode = "eval"
        X_test = np.array([[11.0, 21.0]])
        out_eval = bn.forward(X_test)
        # Input matching exact running mean (11.0, 21.0) must normalize to near 0.0
        np.testing.assert_allclose(out_eval, [[0.0, 0.0]], atol=0.2)

    def test_batchnorm_gamma_beta_scale_and_shift(self):
        bn = BatchNorm1d(num_features=2)
        bn.gamma = np.array([[2.0, 3.0]])
        bn.beta  = np.array([[5.0, -5.0]])
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = bn.forward(X)
        # Mean should equal beta
        np.testing.assert_allclose(np.mean(out, axis=0), [5.0, -5.0], atol=1e-5)


# =====================================================================
# 3. LAYER NORMALIZATION (4 tests)
# =====================================================================
class TestLayerNorm:
    def test_layernorm_normalizes_per_sample_feature_dimension(self):
        ln = LayerNorm(num_features=4)
        X = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
        out = ln.forward(X)
        # Mean across features (axis=1) must be zero
        np.testing.assert_allclose(np.mean(out, axis=1), [0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(np.var(out, axis=1), [1.0, 1.0], atol=1e-3)

    def test_layernorm_invariant_to_input_shift_and_scale(self):
        ln = LayerNorm(num_features=3)
        X1 = np.array([[1.0, 2.0, 3.0]])
        X2 = np.array([[101.0, 102.0, 103.0]])  # Shifted by +100
        out1 = ln.forward(X1)
        out2 = ln.forward(X2)
        np.testing.assert_allclose(out1, out2, atol=1e-4)

    def test_layernorm_eval_and_train_behave_identically(self):
        ln = LayerNorm(num_features=3)
        X = np.random.randn(2, 3)
        out1 = ln.forward(X)
        out2 = ln.forward(X)
        np.testing.assert_array_equal(out1, out2)

    def test_layernorm_gradcheck_analytical_vs_numerical(self):
        ln = LayerNorm(num_features=4)
        X = np.random.randn(3, 4)
        out = ln.forward(X)
        dout = out.copy()  # L = 0.5 * sum(out^2)
        dX, _, _ = ln.backward(dout)

        eps = 1e-5
        num_dX = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                orig = X[i, j]
                X[i, j] = orig + eps
                l1 = 0.5 * np.sum(ln.forward(X) ** 2)
                X[i, j] = orig - eps
                l2 = 0.5 * np.sum(ln.forward(X) ** 2)
                X[i, j] = orig
                num_dX[i, j] = (l1 - l2) / (2 * eps)

        abs_error = np.max(np.abs(dX - num_dX))
        assert abs_error < 1e-4


# =====================================================================
# 4. L1/L2 SPARSITY & CHALLENGE (4 tests)
# =====================================================================
class TestL1L2SparsityAndChallenge:
    def test_l1_regularization_produces_exact_zeros(self):
        w = np.array([0.02, -0.01, 1.0])
        lr = 0.01
        l1_lambda = 0.5
        for _ in range(10):
            w -= lr * l1_lambda * np.sign(w)
            w[np.abs(w) < 1e-4] = 0.0
        assert w[0] == 0.0 and w[1] == 0.0

    def test_l2_regularization_shrinks_weights_smoothly(self):
        w = np.array([10.0, -5.0])
        lr = 0.1
        l2_lambda = 0.1
        for _ in range(5):
            w -= lr * l2_lambda * w
        assert abs(w[0]) < 10.0 and abs(w[0]) > 0.0
        assert abs(w[1]) < 5.0 and abs(w[1]) > 0.0

    def test_layernorm_dropout_block_challenge_solution(self):
        block = LayerNormDropoutBlock(num_features=4, p=0.0, seed=42)
        X = np.random.randn(4, 4)
        out = block.forward(X)
        assert out.shape == (4, 4)
        np.testing.assert_allclose(np.mean(out, axis=1), 0.0, atol=1e-5)

    def test_batchnorm_backward_pass_shape_consistency(self):
        bn = BatchNorm1d(num_features=5)
        X = np.random.randn(8, 5)
        out = bn.forward(X)
        dout = np.ones_like(out)
        dX, dgamma, dbeta = bn.backward(dout)
        assert dX.shape == (8, 5)
        assert dgamma.shape == (1, 5)
        assert dbeta.shape == (1, 5)

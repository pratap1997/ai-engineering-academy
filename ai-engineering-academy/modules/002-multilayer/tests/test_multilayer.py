"""
AI ENGINEERING ACADEMY — MODULE 002 TEST SUITE
Comprehensive Pytest Suite for Multilayer Perceptron & Hidden Layers (16 Tests)
"""

import importlib.util
import os
import sys
import numpy as np
import pytest

# Load Module 002 Implementation
_script_dir = os.path.dirname(os.path.abspath(__file__))
_mod2_dir = os.path.dirname(_script_dir)
_spec = importlib.util.spec_from_file_location(
    "implementation_mod2",
    os.path.join(_mod2_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MultilayerPerceptronPython = _mod.MultilayerPerceptronPython
MultilayerPerceptronNumPy = _mod.MultilayerPerceptronNumPy
make_xor_mlp = _mod.make_xor_mlp

# Load Challenge Solution
_spec_ch = importlib.util.spec_from_file_location(
    "challenge_mod2",
    os.path.join(_mod2_dir, "07-challenge-solution.py"),
)
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)

Parity3InputMLP = _mod_ch.Parity3InputMLP


# =====================================================================
# 1. CORE BEHAVIOUR (4 tests)
# =====================================================================
class TestCoreBehaviour:
    def test_xor_solved_with_handcrafted_weights(self):
        mlp = make_xor_mlp()
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        preds = mlp.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_and_gate_mlp_forward(self):
        # 2-layer MLP for AND gate
        W1 = [[1.0, 1.0]]
        b1 = [-1.5]
        W2 = [[1.0]]
        b2 = [-0.5]
        mlp = MultilayerPerceptronNumPy(W1, b1, W2, b2, "step", "step")
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]])
        np.testing.assert_array_equal(mlp.predict(X), y)

    def test_or_gate_mlp_forward(self):
        W1 = [[1.0, 1.0]]
        b1 = [-0.5]
        W2 = [[1.0]]
        b2 = [-0.5]
        mlp = MultilayerPerceptronNumPy(W1, b1, W2, b2, "step", "step")
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [1]])
        np.testing.assert_array_equal(mlp.predict(X), y)

    def test_output_shape_matches_batch_size(self):
        mlp = make_xor_mlp()
        X_batch = np.random.randn(25, 2)
        preds = mlp.predict(X_batch)
        assert preds.shape == (25, 1)


# =====================================================================
# 2. ACTIVATION FUNCTIONS (4 tests)
# =====================================================================
class TestActivationFunctions:
    def test_step_activation(self):
        W1 = [[1.0, 1.0]]
        b1 = [0.0]
        W2 = [[1.0]]
        b2 = [0.0]
        mlp = MultilayerPerceptronNumPy(W1, b1, W2, b2, "step", "step")
        res = mlp.forward(np.array([[-1, -1], [0, 0], [1, 1]]))
        np.testing.assert_array_equal(res["A1"], np.array([[0], [1], [1]]))

    def test_sigmoid_activation_bounds(self):
        W1 = [[1.0, 1.0]]
        b1 = [0.0]
        W2 = [[1.0]]
        b2 = [0.0]
        mlp = MultilayerPerceptronNumPy(W1, b1, W2, b2, "sigmoid", "sigmoid")
        res = mlp.forward(np.array([[-10, -10], [0, 0], [10, 10]]))
        assert np.all(res["A2"] >= 0.0) and np.all(res["A2"] <= 1.0)

    def test_relu_activation_non_negative(self):
        W1 = [[1.0, -2.0]]
        b1 = [-1.0]
        W2 = [[1.0]]
        b2 = [0.0]
        mlp = MultilayerPerceptronNumPy(W1, b1, W2, b2, "relu", "relu")
        res = mlp.forward(np.array([[0, 5], [5, 0]]))
        assert np.all(res["A1"] >= 0.0)
        assert res["A1"][0, 0] == 0.0  # max(0, 0 - 10 - 1) = 0
        assert res["A1"][1, 0] == 4.0  # max(0, 5 - 0 - 1) = 4

    def test_tanh_activation_range(self):
        W1 = [[1.0, 1.0]]
        b1 = [0.0]
        W2 = [[1.0]]
        b2 = [0.0]
        mlp = MultilayerPerceptronNumPy(W1, b1, W2, b2, "tanh", "tanh")
        res = mlp.forward(np.array([[-5, -5], [5, 5]]))
        assert np.all(res["A2"] >= -1.0) and np.all(res["A2"] <= 1.0)


# =====================================================================
# 3. INPUT VALIDATION & COMPARISON (4 tests)
# =====================================================================
class TestInputValidation:
    def test_invalid_activation_name_raises_key_error(self):
        with pytest.raises(KeyError):
            MultilayerPerceptronNumPy([[1, 1]], [0], [[1]], [0], "invalid_act", "step")

    def test_pure_python_and_numpy_forward_pass_equality(self):
        W1 = [[0.5, -0.5], [1.0, 0.5]]
        b1 = [-0.2, 0.1]
        W2 = [[0.8, -0.4]]
        b2 = [0.1]

        py_mlp = MultilayerPerceptronPython(W1, b1, W2, b2)
        np_mlp = MultilayerPerceptronNumPy(W1, b1, W2, b2)

        X_test = [[0, 0], [0, 1], [1, 0], [1, 1]]
        py_preds = py_mlp.predict(X_test)
        np_preds = np_mlp.predict(np.array(X_test)).tolist()

        assert py_preds == np_preds

    def test_1d_input_automatically_promoted_to_2d(self):
        mlp = make_xor_mlp()
        pred_1d = mlp.predict([1, 0])
        assert pred_1d.shape == (1, 1)
        assert pred_1d[0, 0] == 1.0

    def test_batch_prediction_consistency(self):
        mlp = make_xor_mlp()
        x1 = np.array([1, 0])
        x2 = np.array([0, 1])
        batch = np.array([[1, 0], [0, 1]])

        p1 = mlp.predict(x1)
        p2 = mlp.predict(x2)
        p_batch = mlp.predict(batch)

        assert p_batch[0, 0] == p1[0, 0]
        assert p_batch[1, 0] == p2[0, 0]


# =====================================================================
# 4. LIMITATION & ADVANCED CHALLENGE (4 tests)
# =====================================================================
class TestLimitationAndCollapse:
    def test_linear_layers_collapse_to_single_matrix(self):
        W1 = np.array([[0.5, 0.2], [-0.3, 0.8]])
        b1 = np.array([0.1, -0.2])
        W2 = np.array([[0.7, -0.5]])
        b2 = np.array([0.3])

        X = np.array([[1.0, 2.0], [-1.0, 0.5]])

        # 2 linear layers
        A1 = np.dot(X, W1.T) + b1
        y_2layer = np.dot(A1, W2.T) + b2

        # Collapsed single layer
        W_comb = np.dot(W2, W1)
        b_comb = np.dot(W2, b1) + b2
        y_collapsed = np.dot(X, W_comb.T) + b_comb

        np.testing.assert_allclose(y_2layer, y_collapsed, rtol=1e-5)

    def test_parity_3input_mlp_solves_all_8_cases(self):
        mlp = Parity3InputMLP()
        X_parity = np.array([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ])
        y_parity = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        preds = mlp.predict(X_parity)
        np.testing.assert_array_equal(preds, y_parity)

    def test_zero_weights_yield_bias_only_activations(self):
        W1 = [[0.0, 0.0]]
        b1 = [0.5]
        W2 = [[0.0]]
        b2 = [-0.1]
        mlp = MultilayerPerceptronNumPy(W1, b1, W2, b2, "step", "step")
        X = np.array([[10, 20], [-100, -200]])
        res = mlp.forward(X)
        np.testing.assert_array_equal(res["A1"], np.array([[1], [1]]))
        np.testing.assert_array_equal(res["A2"], np.array([[0], [0]]))

    def test_hidden_space_warping_for_xor(self):
        mlp = make_xor_mlp()
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        res = mlp.forward(X)
        A1 = res["A1"]
        # Expected hidden mapping: (0,0)->(0,1), (0,1)->(1,1), (1,0)->(1,1), (1,1)->(1,0)
        expected_A1 = np.array([[0, 1], [1, 1], [1, 1], [1, 0]])
        np.testing.assert_array_equal(A1, expected_A1)

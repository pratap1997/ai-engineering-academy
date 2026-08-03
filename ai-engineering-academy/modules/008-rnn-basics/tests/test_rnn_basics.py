"""
AI ENGINEERING ACADEMY — MODULE 008 TEST SUITE
Comprehensive Pytest Suite for Recurrent Neural Networks & BPTT (16 Tests)
"""

import importlib.util
import os
import sys
import numpy as np
import pytest

# Load Module 008 Implementation
_script_dir = os.path.dirname(os.path.abspath(__file__))
_mod8_dir = os.path.dirname(_script_dir)
_spec = importlib.util.spec_from_file_location(
    "implementation_mod8",
    os.path.join(_mod8_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

RNNCell = _mod.RNNCell
RNNSequence = _mod.RNNSequence
GradientClipper = _mod.GradientClipper

# Load Challenge Solution
_spec_ch = importlib.util.spec_from_file_location(
    "challenge_mod8",
    os.path.join(_mod8_dir, "07-challenge-solution.py"),
)
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)

verify_rnn_bptt_gradcheck = _mod_ch.verify_rnn_bptt_gradcheck


# =====================================================================
# 1. RNN CELL FORWARD (4 tests)
# =====================================================================
class TestRNNCellForward:
    def test_rnn_cell_forward_output_shape(self):
        cell = RNNCell(in_features=3, hidden_features=5, seed=42)
        x = np.random.randn(2, 3)
        h_prev = np.random.randn(2, 5)
        h_next, _ = cell.forward(x, h_prev)
        assert h_next.shape == (2, 5)

    def test_rnn_cell_tanh_activation_bounds(self):
        cell = RNNCell(in_features=3, hidden_features=5, seed=42)
        cell.W_xh.fill(10.0)
        cell.W_hh.fill(10.0)
        x = np.ones((1, 3))
        h_prev = np.ones((1, 5))
        h_next, _ = cell.forward(x, h_prev)
        assert np.all(h_next >= -1.0) and np.all(h_next <= 1.0)

    def test_rnn_cell_zero_weights_yield_zero_output(self):
        cell = RNNCell(in_features=3, hidden_features=5)
        cell.W_xh.fill(0.0)
        cell.W_hh.fill(0.0)
        cell.b_h.fill(0.0)
        x = np.random.randn(1, 3)
        h_prev = np.random.randn(1, 5)
        h_next, _ = cell.forward(x, h_prev)
        np.testing.assert_allclose(h_next, 0.0)

    def test_rnn_cell_different_inputs_produce_different_states(self):
        cell = RNNCell(in_features=2, hidden_features=4, seed=42)
        h0 = np.zeros((1, 4))
        h1, _ = cell.forward(np.array([[1.0, 0.0]]), h0)
        h2, _ = cell.forward(np.array([[0.0, 1.0]]), h0)
        assert not np.allclose(h1, h2)


# =====================================================================
# 2. RNN SEQUENCE UNROLLED (4 tests)
# =====================================================================
class TestRNNSequenceUnrolled:
    def test_rnn_sequence_unrolled_output_shape_matches_T(self):
        rnn = RNNSequence(in_features=4, hidden_features=8, out_features=3, seed=42)
        X = np.random.randn(2, 6, 4)  # N=2, T=6, D=4
        Y = rnn.forward(X)
        assert Y.shape == (2, 6, 3)

    def test_rnn_sequence_hidden_state_history_length(self):
        rnn = RNNSequence(in_features=2, hidden_features=4, out_features=1, seed=42)
        X = np.random.randn(1, 5, 2)
        rnn.forward(X)
        assert len(rnn.h_states) == 6  # h0 + h1..h5 = 6

    def test_rnn_sequence_h0_custom_initialization(self):
        rnn = RNNSequence(in_features=2, hidden_features=4, out_features=1, seed=42)
        X = np.random.randn(1, 3, 2)
        h0 = np.ones((1, 4)) * 0.5
        rnn.forward(X, h0=h0)
        np.testing.assert_allclose(rnn.h_states[0], h0)

    def test_rnn_sequence_single_step_matches_cell(self):
        rnn = RNNSequence(in_features=2, hidden_features=4, out_features=1, seed=42)
        X = np.random.randn(1, 1, 2)
        Y_seq = rnn.forward(X)
        x1 = X[:, 0, :]
        h1, _ = rnn.cell.forward(x1, np.zeros((1, 4)))
        y1 = np.dot(h1, rnn.W_hy) + rnn.b_y
        np.testing.assert_allclose(Y_seq[:, 0, :], y1)


# =====================================================================
# 3. BPTT & GRADIENT CLIPPING (4 tests)
# =====================================================================
class TestBPTTAndClipping:
    def test_bptt_backward_gradient_shapes(self):
        rnn = RNNSequence(in_features=3, hidden_features=5, out_features=2, seed=42)
        X = np.random.randn(2, 4, 3)
        Y = rnn.forward(X)
        dY = np.ones_like(Y)
        dX, grads = rnn.backward(dY)

        assert dX.shape == (2, 4, 3)
        assert grads["dW_xh"].shape == (3, 5)
        assert grads["dW_hh"].shape == (5, 5)
        assert grads["dW_hy"].shape == (5, 2)

    def test_gradient_clipper_caps_norm_at_threshold(self):
        clipper = GradientClipper(max_norm=1.0)
        grads = {"dW": np.array([[10.0, 20.0], [30.0, 40.0]])}
        clipped_grads, norm = clipper.clip(grads)
        clipped_norm = np.sqrt(np.sum(clipped_grads["dW"] ** 2))
        assert abs(clipped_norm - 1.0) < 1e-5

    def test_gradient_clipper_does_not_modify_small_gradients(self):
        clipper = GradientClipper(max_norm=5.0)
        grads = {"dW": np.array([[0.1, 0.2]])}
        orig_dW = grads["dW"].copy()
        clipped_grads, _ = clipper.clip(grads)
        np.testing.assert_allclose(clipped_grads["dW"], orig_dW)

    def test_bptt_gradcheck_analytical_vs_numerical(self):
        rnn = RNNSequence(in_features=2, hidden_features=3, out_features=1, seed=42)
        X = np.random.randn(2, 3, 2)
        Y = rnn.forward(X)
        dY = Y.copy()
        dX_ana, grads_ana = rnn.backward(dY)

        eps = 1e-5
        num_dW_hh = np.zeros_like(rnn.cell.W_hh)
        for i in range(rnn.cell.W_hh.shape[0]):
            for j in range(rnn.cell.W_hh.shape[1]):
                orig = rnn.cell.W_hh[i, j]
                rnn.cell.W_hh[i, j] = orig + eps
                l1 = 0.5 * np.sum(rnn.forward(X) ** 2)
                rnn.cell.W_hh[i, j] = orig - eps
                l2 = 0.5 * np.sum(rnn.forward(X) ** 2)
                rnn.cell.W_hh[i, j] = orig
                num_dW_hh[i, j] = (l1 - l2) / (2 * eps)

        abs_error = np.max(np.abs(grads_ana["dW_hh"] - num_dW_hh))
        assert abs_error < 1e-4


# =====================================================================
# 4. END-TO-END RNN (4 tests)
# =====================================================================
class TestEndToEndRNN:
    def test_exploding_gradient_norm_increases_with_T(self):
        rnn = RNNSequence(in_features=2, hidden_features=4, out_features=1, seed=42)
        rnn.cell.W_hh = np.eye(4) * 2.0  # Eigenvalue 2.0

        X5 = np.random.randn(1, 5, 2)
        rnn.forward(X5)
        _, grads5 = rnn.backward(np.ones((1, 5, 1)))
        norm5 = np.sqrt(np.sum(grads5["dW_hh"] ** 2))

        X15 = np.random.randn(1, 15, 2)
        rnn.forward(X15)
        _, grads15 = rnn.backward(np.ones((1, 15, 1)))
        norm15 = np.sqrt(np.sum(grads15["dW_hh"] ** 2))

        assert norm15 > norm5

    def test_rnn_bptt_gradcheck_challenge_solution(self):
        verify_rnn_bptt_gradcheck()

    def test_bptt_accumulates_gradients_across_all_time_steps(self):
        rnn = RNNSequence(in_features=2, hidden_features=4, out_features=1, seed=42)
        X = np.random.randn(1, 4, 2)
        rnn.forward(X)
        _, grads = rnn.backward(np.ones((1, 4, 1)))
        assert np.max(np.abs(grads["dW_hh"])) > 0.0

    def test_rnn_sequence_loss_decreases_over_training(self):
        rnn = RNNSequence(in_features=2, hidden_features=4, out_features=1, seed=42)
        X = np.random.randn(2, 3, 2)
        Y_target = np.random.randn(2, 3, 1)

        lr = 0.05
        initial_loss = 0.5 * np.mean((rnn.forward(X) - Y_target) ** 2)

        for _ in range(20):
            Y_pred = rnn.forward(X)
            dY = (Y_pred - Y_target) / 6.0
            _, grads = rnn.backward(dY)

            rnn.cell.W_xh -= lr * grads["dW_xh"]
            rnn.cell.W_hh -= lr * grads["dW_hh"]
            rnn.cell.b_h  -= lr * grads["db_h"]
            rnn.W_hy      -= lr * grads["dW_hy"]
            rnn.b_y       -= lr * grads["db_y"]

        final_loss = 0.5 * np.mean((rnn.forward(X) - Y_target) ** 2)
        assert final_loss < initial_loss

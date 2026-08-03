"""
AI ENGINEERING ACADEMY -- MODULE 022 TEST SUITE
Comprehensive Pytest Suite for Knowledge Distillation (16 Tests)
"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_mod22_dir = os.path.dirname(_dir)

_spec = importlib.util.spec_from_file_location("impl_mod22", os.path.join(_mod22_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

softmax_temperature     = _mod.softmax_temperature
kl_divergence_loss      = _mod.kl_divergence_loss
cross_entropy_loss      = _mod.cross_entropy_loss
distillation_loss       = _mod.distillation_loss
ToyLinearModel          = _mod.ToyLinearModel
train_distillation_step = _mod.train_distillation_step

_spec_ch = importlib.util.spec_from_file_location("ch_mod22", os.path.join(_mod22_dir, "07-challenge-solution.py"))
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)
MultiLayerStudentDistiller = _mod_ch.MultiLayerStudentDistiller
verify_multi_layer_student_distillation = _mod_ch.verify_multi_layer_student_distillation


# ===================================================================
# 1. TEMPERATURE SOFTMAX (4 tests)
# ===================================================================
class TestTemperatureSoftmax:
    def test_softmax_temp_1_matches_standard(self):
        logits = np.array([[2.0, 1.0, 0.0]])
        p1 = softmax_temperature(logits, temperature=1.0)
        p_std = np.exp(logits) / np.sum(np.exp(logits))
        np.testing.assert_allclose(p1, p_std, atol=1e-6)

    def test_softmax_higher_temp_flattens(self):
        logits = np.array([[5.0, 1.0, 0.0]])
        p1 = softmax_temperature(logits, temperature=1.0)
        p4 = softmax_temperature(logits, temperature=4.0)
        assert p4[0, 0] < p1[0, 0]
        assert p4[0, 1] > p1[0, 1]

    def test_softmax_temp_sum_to_one(self):
        logits = np.random.randn(3, 5)
        p = softmax_temperature(logits, temperature=3.0)
        row_sums = np.sum(p, axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_softmax_temp_no_nans(self):
        logits = np.random.randn(2, 5) * 100.0
        p = softmax_temperature(logits, temperature=4.0)
        assert not np.isnan(p).any()


# ===================================================================
# 2. KL DIVERGENCE & DISTILLATION LOSS (4 tests)
# ===================================================================
class TestDistillationLoss:
    def test_kl_zero_for_identical_distributions(self):
        p = np.array([[0.5, 0.3, 0.2]])
        kl = kl_divergence_loss(p, p)
        np.testing.assert_allclose(kl, 0.0, atol=1e-6)

    def test_kl_positive_for_different_distributions(self):
        p1 = np.array([[0.7, 0.2, 0.1]])
        p2 = np.array([[0.3, 0.4, 0.3]])
        kl = kl_divergence_loss(p1, p2)
        assert kl > 0.0

    def test_distillation_loss_combines_kl_and_ce(self):
        t_logits = np.array([[2.0, 1.0]])
        s_logits = np.array([[1.0, 2.0]])
        y_onehot = np.array([[1.0, 0.0]])

        total, kl, ce = distillation_loss(t_logits, s_logits, y_onehot, temperature=2.0, alpha=0.5)
        expected = 0.5 * (4.0) * kl + 0.5 * ce
        np.testing.assert_allclose(total, expected, atol=1e-5)

    def test_distillation_loss_no_nans(self):
        t_logits = np.random.randn(4, 5)
        s_logits = np.random.randn(4, 5)
        y_onehot = np.eye(5)[np.array([0, 1, 2, 3])]
        total, kl, ce = distillation_loss(t_logits, s_logits, y_onehot, temperature=4.0, alpha=0.7)
        assert not np.isnan(total)


# ===================================================================
# 3. DISTILLATION TRAINING STEP (4 tests)
# ===================================================================
class TestDistillationTrainingStep:
    def test_training_step_reduces_distill_loss(self):
        teacher = ToyLinearModel(16, 5, seed=101)
        student = ToyLinearModel(16, 5, seed=202)
        x = np.random.randn(4, 16)
        y_onehot = np.eye(5)[np.array([0, 1, 2, 3])]

        loss_0, _, _ = train_distillation_step(student, teacher, x, y_onehot, temperature=4.0, alpha=0.7, lr=0.0)
        for _ in range(5):
            loss_step, _, _ = train_distillation_step(student, teacher, x, y_onehot, temperature=4.0, alpha=0.7, lr=0.1)

        assert loss_step < loss_0

    def test_teacher_weights_remain_unchanged(self):
        teacher = ToyLinearModel(16, 5, seed=101)
        w_orig = teacher.W.copy()
        student = ToyLinearModel(16, 5, seed=202)
        x = np.random.randn(4, 16)
        y_onehot = np.eye(5)[np.array([0, 1, 2, 3])]

        train_distillation_step(student, teacher, x, y_onehot, temperature=4.0, alpha=0.7, lr=0.1)
        np.testing.assert_allclose(teacher.W, w_orig)

    def test_student_weights_updated(self):
        teacher = ToyLinearModel(16, 5, seed=101)
        student = ToyLinearModel(16, 5, seed=202)
        w_orig = student.W.copy()
        x = np.random.randn(4, 16)
        y_onehot = np.eye(5)[np.array([0, 1, 2, 3])]

        train_distillation_step(student, teacher, x, y_onehot, temperature=4.0, alpha=0.7, lr=0.1)
        assert not np.allclose(student.W, w_orig)

    def test_kl_divergence_decreases(self):
        teacher = ToyLinearModel(16, 5, seed=101)
        student = ToyLinearModel(16, 5, seed=202)
        x = np.random.randn(4, 16)
        y_onehot = np.eye(5)[np.array([0, 1, 2, 3])]

        _, kl_0, _ = train_distillation_step(student, teacher, x, y_onehot, temperature=4.0, alpha=0.7, lr=0.0)
        for _ in range(5):
            _, kl_step, _ = train_distillation_step(student, teacher, x, y_onehot, temperature=4.0, alpha=0.7, lr=0.1)

        assert kl_step < kl_0


# ===================================================================
# 4. CHALLENGE VERIFICATION (4 tests)
# ===================================================================
class TestDistillationChallenge:
    def test_challenge_verification_runs(self):
        verify_multi_layer_student_distillation()

    def test_multi_layer_distiller_kl_decreases(self):
        distiller = MultiLayerStudentDistiller(in_dim=16, num_classes=5, seed_t=101, seed_s=202)
        x = np.random.randn(4, 16)
        y_onehot = np.eye(5)[np.array([0, 1, 2, 3])]

        _, kl_0, _ = distiller.train_step(x, y_onehot, temperature=4.0, alpha=0.7, lr=0.1)
        for _ in range(10):
            _, kl_step, _ = distiller.train_step(x, y_onehot, temperature=4.0, alpha=0.7, lr=0.1)

        assert kl_step < kl_0

    def test_toy_model_output_shape(self):
        m = ToyLinearModel(16, 5, seed=42)
        x = np.random.randn(3, 16)
        out = m.forward(x)
        assert out.shape == (3, 5)

    def test_cross_entropy_non_negative(self):
        y = np.array([[1.0, 0.0], [0.0, 1.0]])
        p = np.array([[0.8, 0.2], [0.1, 0.9]])
        ce = cross_entropy_loss(y, p)
        assert ce > 0.0

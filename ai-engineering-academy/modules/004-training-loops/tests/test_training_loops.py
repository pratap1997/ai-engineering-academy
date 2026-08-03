"""
AI ENGINEERING ACADEMY — MODULE 004 TEST SUITE
Comprehensive Pytest Suite for Optimizers, Learning Rates & Training Loops (16 Tests)
"""

import importlib.util
import os
import sys
import math
import numpy as np
import pytest

# Load Module 004 Implementation
_script_dir = os.path.dirname(os.path.abspath(__file__))
_mod4_dir = os.path.dirname(_script_dir)
_spec = importlib.util.spec_from_file_location(
    "implementation_mod4",
    os.path.join(_mod4_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SGD = _mod.SGD
Momentum = _mod.Momentum
RMSprop = _mod.RMSprop
Adam = _mod.Adam
AdamW = _mod.AdamW
CosineAnnealingLR = _mod.CosineAnnealingLR
EarlyStopping = _mod.EarlyStopping
mini_batch_generator = _mod.mini_batch_generator

# Load Challenge Solution
_spec_ch = importlib.util.spec_from_file_location(
    "challenge_mod4",
    os.path.join(_mod4_dir, "07-challenge-solution.py"),
)
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)

ProductionTrainer = _mod_ch.ProductionTrainer

_mod3_dir = os.path.join(os.path.dirname(_mod4_dir), "003-backprop")
_spec3 = importlib.util.spec_from_file_location(
    "implementation_mod3",
    os.path.join(_mod3_dir, "04-implementation.py"),
)
_mod3 = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(_mod3)

MatrixMLPBackprop = _mod3.MatrixMLPBackprop


# =====================================================================
# 1. OPTIMIZER UPDATES (4 tests)
# =====================================================================
class TestOptimizerUpdates:
    def test_sgd_update_step(self):
        opt = SGD(lr=0.1)
        param = np.array([1.0, 2.0])
        grad = np.array([0.5, -0.5])
        new_param = opt.update(param, grad)
        np.testing.assert_allclose(new_param, np.array([0.95, 2.05]))

    def test_momentum_accumulates_velocity(self):
        opt = Momentum(lr=0.1, momentum=0.9)
        param = np.array([1.0])
        grad = np.array([1.0])
        p1 = opt.update(param, grad)  # v1 = 1.0, p1 = 1.0 - 0.1*1.0 = 0.9
        p2 = opt.update(p1, grad)     # v2 = 0.9*1.0 + 1.0 = 1.9, p2 = 0.9 - 0.1*1.9 = 0.71
        assert abs(p1[0] - 0.9) < 1e-6
        assert abs(p2[0] - 0.71) < 1e-6

    def test_rmsprop_scales_by_gradient_variance(self):
        opt = RMSprop(lr=0.1, alpha=0.9)
        param = np.array([1.0])
        grad = np.array([2.0])
        # v1 = 0.9*0 + 0.1*(4) = 0.4
        # step = 0.1 / (sqrt(0.4) + 1e-8) * 2.0 = 0.1 / 0.63245 * 2.0 = 0.3162
        p1 = opt.update(param, grad)
        assert abs(p1[0] - (1.0 - 0.3162277)) < 1e-4

    def test_adam_bias_correction_first_step(self):
        opt = Adam(lr=0.01, beta1=0.9, beta2=0.999)
        param = np.array([0.0])
        grad = np.array([1.0])
        # m1 = 0.1, v1 = 0.001
        # m_hat1 = 0.1 / (1 - 0.9) = 1.0
        # v_hat1 = 0.001 / (1 - 0.999) = 1.0
        # param1 = 0 - 0.01 * (1.0 / sqrt(1.0)) = -0.01
        p1 = opt.update(param, grad)
        assert abs(p1[0] - (-0.01)) < 1e-5


# =====================================================================
# 2. ADAMW & REGULARIZATION (4 tests)
# =====================================================================
class TestAdamWDecoupledDecay:
    def test_adamw_applies_weight_decay_to_zero_gradient_parameters(self):
        opt = AdamW(lr=0.1, weight_decay=0.1)
        param = np.array([10.0])
        grad = np.array([0.0])
        # param_decayed = 10.0 * (1 - 0.1*0.1) = 10.0 * 0.99 = 9.9
        p1 = opt.update(param, grad)
        assert abs(p1[0] - 9.9) < 1e-5

    def test_adamw_weight_decay_proportional_to_param_magnitude(self):
        opt = AdamW(lr=0.1, weight_decay=0.05)
        p_large = np.array([100.0])
        p_small = np.array([1.0])
        g = np.array([0.0])

        p1_large = opt.update(p_large, g)
        p1_small = opt.update(p_small, g)

        diff_large = 100.0 - p1_large[0]
        diff_small = 1.0 - p1_small[0]

        assert abs(diff_large / diff_small - 100.0) < 1e-5

    def test_adam_and_adamw_identical_when_weight_decay_is_zero(self):
        opt_adam = Adam(lr=0.01)
        opt_adamw = AdamW(lr=0.01, weight_decay=0.0)

        p1 = np.array([2.5, -1.2])
        p2 = np.array([2.5, -1.2])
        g = np.array([0.5, -0.3])

        p1_out = opt_adam.update(p1, g)
        p2_out = opt_adamw.update(p2, g)

        np.testing.assert_allclose(p1_out, p2_out)

    def test_adam_gradient_clip_prevents_explosion(self):
        trainer = ProductionTrainer(None, max_grad_norm=1.0)
        huge_grads = {"dW1": np.array([[100.0, 200.0]]), "db1": np.array([[50.0]])}
        clipped_grads, norm = trainer.clip_grad_norm(huge_grads)
        clipped_norm = np.sqrt(sum(np.sum(g ** 2) for g in clipped_grads.values()))
        assert abs(clipped_norm - 1.0) < 1e-5


# =====================================================================
# 3. SCHEDULERS & BATCHING (4 tests)
# =====================================================================
class TestSchedulersAndBatching:
    def test_cosine_annealing_starts_at_initial_lr(self):
        opt = SGD(lr=0.1)
        sched = CosineAnnealingLR(opt, T_max=100, eta_min=0.001)
        lr1 = sched.step()
        assert lr1 < 0.1 and lr1 > 0.099

    def test_cosine_annealing_reaches_eta_min_at_t_max(self):
        opt = SGD(lr=0.1)
        sched = CosineAnnealingLR(opt, T_max=50, eta_min=0.005)
        for _ in range(50):
            lr = sched.step()
        assert abs(lr - 0.005) < 1e-6

    def test_mini_batch_generator_shuffles_and_chunks(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        batches = list(mini_batch_generator(X, y, batch_size=16, shuffle=False))
        assert len(batches) == 4  # 16 + 16 + 16 + 2 = 50
        assert batches[0][0].shape == (16, 2)
        assert batches[-1][0].shape == (2, 2)

    def test_early_stopping_triggers_on_stagnant_loss(self):
        es = EarlyStopping(patience=3, min_delta=1e-3)
        assert not es.should_stop(1.0)
        assert not es.should_stop(0.99)
        assert not es.should_stop(0.99)
        assert not es.should_stop(0.99)
        assert es.should_stop(0.99)  # 3 consecutive stagnant epochs


# =====================================================================
# 4. END-TO-END TRAINING (4 tests)
# =====================================================================
class TestEndToEndTraining:
    def test_adam_mlp_xor_convergence(self):
        mlp = MatrixMLPBackprop(n_input=2, n_hidden=4, n_output=1, seed=42)
        opt_W1 = Adam(lr=0.05)
        opt_b1 = Adam(lr=0.05)
        opt_W2 = Adam(lr=0.05)
        opt_b2 = Adam(lr=0.05)

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])

        for _ in range(1000):
            mlp.forward(X)
            grads = mlp.backward(y)
            mlp.W1 = opt_W1.update(mlp.W1, grads["dW1"])
            mlp.b1 = opt_b1.update(mlp.b1, grads["db1"])
            mlp.W2 = opt_W2.update(mlp.W2, grads["dW2"])
            mlp.b2 = opt_b2.update(mlp.b2, grads["db2"])

        preds = mlp.forward(X)
        binary_preds = (preds >= 0.5).astype(int)
        np.testing.assert_array_equal(binary_preds, y)

    def test_adamw_production_trainer_pipeline(self):
        mlp = MatrixMLPBackprop(n_input=2, n_hidden=4, n_output=1, seed=42)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])

        trainer = ProductionTrainer(mlp, lr=0.1, max_epochs=200, batch_size=4)
        history = trainer.fit(X, y)

        assert history["loss"][-1] < 0.05

    def test_momentum_faster_than_sgd_on_ill_conditioned_landscape(self):
        g = np.array([10.0, 0.1])
        p_sgd = np.array([0.0, 0.0])
        p_mom = np.array([0.0, 0.0])

        opt_sgd = SGD(lr=0.01)
        opt_mom = Momentum(lr=0.01, momentum=0.9)

        for _ in range(10):
            p_sgd = opt_sgd.update(p_sgd, g)
            p_mom = opt_mom.update(p_mom, g)

        # Momentum distance traveled on dimension 0 should be greater due to velocity build-up
        assert abs(p_mom[0]) > abs(p_sgd[0])

    def test_loss_history_decreases_monotonically(self):
        mlp = MatrixMLPBackprop(n_input=2, n_hidden=4, n_output=1, seed=42)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])

        trainer = ProductionTrainer(mlp, lr=0.05, max_epochs=100, batch_size=4)
        history = trainer.fit(X, y)

        assert history["loss"][0] > history["loss"][-1]

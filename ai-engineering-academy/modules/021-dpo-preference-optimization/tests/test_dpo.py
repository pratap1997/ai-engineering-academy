"""
AI ENGINEERING ACADEMY -- MODULE 021 TEST SUITE
Comprehensive Pytest Suite for Direct Preference Optimization (DPO) (16 Tests)
"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_mod21_dir = os.path.dirname(_dir)

_spec = importlib.util.spec_from_file_location("impl_mod21", os.path.join(_mod21_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_implicit_rewards = _mod.compute_implicit_rewards
dpo_loss                 = _mod.dpo_loss
SimplePolicyModel        = _mod.SimplePolicyModel
train_dpo_step           = _mod.train_dpo_step

_spec_ch = importlib.util.spec_from_file_location("ch_mod21", os.path.join(_mod21_dir, "07-challenge-solution.py"))
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)
DPOAlignmentTrainer = _mod_ch.DPOAlignmentTrainer
verify_dpo_alignment_trainer = _mod_ch.verify_dpo_alignment_trainer


# ===================================================================
# 1. IMPLICIT REWARD COMPUTATION (4 tests)
# ===================================================================
class TestImplicitReward:
    def test_reward_zero_when_policy_equals_ref(self):
        pol = np.array([-1.0, -2.0, -0.5])
        ref = np.array([-1.0, -2.0, -0.5])
        rewards = compute_implicit_rewards(pol, ref, beta=0.1)
        np.testing.assert_allclose(rewards, 0.0)

    def test_reward_positive_when_policy_higher(self):
        pol = np.array([-0.5, -1.0])
        ref = np.array([-1.5, -2.0])
        rewards = compute_implicit_rewards(pol, ref, beta=0.1)
        assert (rewards > 0.0).all()

    def test_reward_scales_with_beta(self):
        pol = np.array([-0.5])
        ref = np.array([-1.5])
        r1 = compute_implicit_rewards(pol, ref, beta=0.1)
        r2 = compute_implicit_rewards(pol, ref, beta=0.5)
        assert abs(r2[0] - 5 * r1[0]) < 1e-6

    def test_reward_shape_maintained(self):
        pol = np.random.randn(4)
        ref = np.random.randn(4)
        rewards = compute_implicit_rewards(pol, ref, beta=0.1)
        assert rewards.shape == (4,)


# ===================================================================
# 2. DPO LOSS & MARGINS (4 tests)
# ===================================================================
class TestDPOLoss:
    def test_dpo_loss_initial_value(self):
        pol_c = np.array([-1.0, -1.5])
        pol_r = np.array([-2.0, -2.5])
        ref_c = pol_c.copy()
        ref_r = pol_r.copy()

        loss, chosen_r, rejected_r, margin = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
        np.testing.assert_allclose(loss, np.log(2.0), atol=1e-5)
        np.testing.assert_allclose(margin, 0.0, atol=1e-5)

    def test_dpo_loss_decreases_when_chosen_reward_higher(self):
        pol_c = np.array([-0.5])  # Increased chosen logp
        pol_r = np.array([-3.0])  # Decreased rejected logp
        ref_c = np.array([-1.5])
        ref_r = np.array([-1.5])

        loss, _, _, margin = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
        assert loss < np.log(2.0)
        assert margin[0] > 0.0

    def test_dpo_loss_no_nans(self):
        pol_c = np.array([-0.1, -10.0])
        pol_r = np.array([-20.0, -0.1])
        ref_c = np.array([-1.0, -1.0])
        ref_r = np.array([-1.0, -1.0])

        loss, _, _, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
        assert not np.isnan(loss)

    def test_reward_margin_calculation(self):
        pol_c = np.array([-0.5])
        pol_r = np.array([-2.0])
        ref_c = np.array([-1.0])
        ref_r = np.array([-1.0])

        _, chosen_r, rejected_r, margin = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
        np.testing.assert_allclose(margin, chosen_r - rejected_r)


# ===================================================================
# 3. DPO TRAINING STEP (4 tests)
# ===================================================================
class TestDPOTrainingStep:
    def test_dpo_step_reduces_loss(self):
        ref_model = SimplePolicyModel(feature_dim=8, seed=42)
        policy_model = ref_model.copy()
        feat_c = np.random.randn(4, 8) + 0.5
        feat_r = np.random.randn(4, 8) - 0.5

        pol_c_0 = policy_model.forward_logps(feat_c)
        pol_r_0 = policy_model.forward_logps(feat_r)
        ref_c_0 = ref_model.forward_logps(feat_c)
        ref_r_0 = ref_model.forward_logps(feat_r)
        loss_0, _, _, _ = dpo_loss(pol_c_0, pol_r_0, ref_c_0, ref_r_0, beta=0.1)

        # Run 5 optimization steps
        for _ in range(5):
            loss_step, _ = train_dpo_step(policy_model, ref_model, feat_c, feat_r, beta=0.1, lr=0.2)

        assert loss_step < loss_0

    def test_reference_model_remains_unchanged(self):
        ref_model = SimplePolicyModel(feature_dim=8, seed=42)
        w_orig = ref_model.w.copy()
        policy_model = ref_model.copy()
        feat_c = np.random.randn(4, 8)
        feat_r = np.random.randn(4, 8)

        train_dpo_step(policy_model, ref_model, feat_c, feat_r, beta=0.1, lr=0.2)
        np.testing.assert_allclose(ref_model.w, w_orig)

    def test_policy_weights_updated(self):
        ref_model = SimplePolicyModel(feature_dim=8, seed=42)
        policy_model = ref_model.copy()
        feat_c = np.random.randn(4, 8) + 0.5
        feat_r = np.random.randn(4, 8) - 0.5

        train_dpo_step(policy_model, ref_model, feat_c, feat_r, beta=0.1, lr=0.2)
        assert not np.allclose(policy_model.w, ref_model.w)

    def test_multiple_dpo_steps_increase_margin(self):
        ref_model = SimplePolicyModel(feature_dim=8, seed=42)
        policy_model = ref_model.copy()
        feat_c = np.random.randn(4, 8) + 0.5
        feat_r = np.random.randn(4, 8) - 0.5

        m0 = 0.0
        for _ in range(5):
            _, m_curr = train_dpo_step(policy_model, ref_model, feat_c, feat_r, beta=0.1, lr=0.1)

        assert np.mean(m_curr) > m0


# ===================================================================
# 4. CHALLENGE VERIFICATION (4 tests)
# ===================================================================
class TestDPOChallenge:
    def test_challenge_verification_runs(self):
        verify_dpo_alignment_trainer()

    def test_trainer_accuracy_increases(self):
        feat_c = np.random.randn(8, 8) + 0.8
        feat_r = np.random.randn(8, 8) - 0.8
        trainer = DPOAlignmentTrainer(feature_dim=8, beta=0.1, lr=0.2, seed=42)
        history = trainer.train_dataset(feat_c, feat_r, epochs=15)
        assert history[-1][2] >= history[0][2]

    def test_policy_model_copy(self):
        m1 = SimplePolicyModel(feature_dim=8, seed=42)
        m2 = m1.copy()
        np.testing.assert_allclose(m1.w, m2.w)
        m2.w += 1.0
        assert not np.allclose(m1.w, m2.w)

    def test_forward_logps_shape(self):
        m = SimplePolicyModel(feature_dim=8, seed=42)
        feat = np.random.randn(5, 8)
        logps = m.forward_logps(feat)
        assert logps.shape == (5,)

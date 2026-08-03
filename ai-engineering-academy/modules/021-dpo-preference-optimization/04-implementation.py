"""
AI ENGINEERING ACADEMY -- MODULE 021
Direct Preference Optimization (DPO) Implementation (Pure Python & NumPy)

Provides:
1. `compute_implicit_rewards`: Calculating r_hat(x, y) = beta * (log pi_theta - log pi_ref).
2. `dpo_loss`: Calculating DPO binary cross-entropy loss & reward margins.
3. `SimplePolicyModel`: Toy logit model for DPO preference training.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def log_sigmoid(x):
    return -np.log1p(np.exp(-np.clip(x, -30.0, 30.0)))


# =====================================================================
# 1. IMPLICIT REWARD & DPO LOSS COMPUTATION
# =====================================================================

def compute_implicit_rewards(policy_logps, ref_logps, beta=0.1):
    """
    policy_logps: (batch_size,) log probabilities under policy pi_theta
    ref_logps:    (batch_size,) log probabilities under reference pi_ref
    Returns implicit_rewards: (batch_size,) beta * (log pi_theta - log pi_ref)
    """
    return beta * (policy_logps - ref_logps)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    Computes DPO Loss and Implicit Rewards.

    Returns:
      loss:           scalar DPO loss value
      chosen_rewards:   implicit rewards for preferred completions
      rejected_rewards: implicit rewards for dispreferred completions
      reward_margins:   chosen_rewards - rejected_rewards
    """
    chosen_rewards = compute_implicit_rewards(policy_chosen_logps, ref_chosen_logps, beta=beta)
    rejected_rewards = compute_implicit_rewards(policy_rejected_logps, ref_rejected_logps, beta=beta)

    logits = chosen_rewards - rejected_rewards
    losses = -log_sigmoid(logits)

    loss = np.mean(losses)
    reward_margins = chosen_rewards - rejected_rewards

    return loss, chosen_rewards, rejected_rewards, reward_margins


# =====================================================================
# 2. TOY POLICY MODEL FOR PREFERENCE OPTIMIZATION
# =====================================================================

class SimplePolicyModel:
    """
    Toy linear model producing log probabilities for chosen and rejected completions.
    """

    def __init__(self, feature_dim=16, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.w = np.random.randn(feature_dim) * 0.1

    def forward_logps(self, features):
        """features: (batch_size, feature_dim) -> logps (batch_size,)"""
        logits = np.matmul(features, self.w)
        # Numerical log-softmax simulation
        return logits - np.log(1.0 + np.exp(logits))

    def copy(self):
        new_model = SimplePolicyModel(feature_dim=len(self.w))
        new_model.w = self.w.copy()
        return new_model


def train_dpo_step(policy, reference, feat_chosen, feat_rejected, beta=0.1, lr=0.1):
    """Executes single gradient update step on policy model using DPO loss."""
    # Current policy logps
    pol_chosen_logps = policy.forward_logps(feat_chosen)
    pol_rejected_logps = policy.forward_logps(feat_rejected)

    # Frozen reference logps
    ref_chosen_logps = reference.forward_logps(feat_chosen)
    ref_rejected_logps = reference.forward_logps(feat_rejected)

    loss, chosen_r, rejected_r, margin = dpo_loss(
        pol_chosen_logps, pol_rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        beta=beta
    )

    # Numerical finite-difference gradient calculation for w
    eps = 1e-5
    grad_w = np.zeros_like(policy.w)
    for i in range(len(policy.w)):
        policy.w[i] += eps
        pol_c_plus = policy.forward_logps(feat_chosen)
        pol_r_plus = policy.forward_logps(feat_rejected)
        l_plus, _, _, _ = dpo_loss(pol_c_plus, pol_r_plus, ref_chosen_logps, ref_rejected_logps, beta=beta)

        policy.w[i] -= 2 * eps
        pol_c_minus = policy.forward_logps(feat_chosen)
        pol_r_minus = policy.forward_logps(feat_rejected)
        l_minus, _, _, _ = dpo_loss(pol_c_minus, pol_r_minus, ref_chosen_logps, ref_rejected_logps, beta=beta)

        policy.w[i] += eps  # Restore
        grad_w[i] = (l_plus - l_minus) / (2 * eps)

    # Update policy weights
    policy.w -= lr * grad_w

    return loss, margin


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 021 -- DIRECT PREFERENCE OPTIMIZATION (DPO) VERIFICATION")
    print("=" * 65)

    np.random.seed(42)
    B, d_feat = 4, 8

    # Reference model
    ref_model = SimplePolicyModel(feature_dim=d_feat, seed=42)
    # Policy model initialized identically at step 0
    policy_model = ref_model.copy()

    feat_chosen = np.random.randn(B, d_feat) + 0.5   # Preferred completions
    feat_rejected = np.random.randn(B, d_feat) - 0.5 # Dispreferred completions

    # Initial loss & margin at step 0 (policy == ref -> margin = 0 -> loss = log(2) ~= 0.6931)
    pol_c_0 = policy_model.forward_logps(feat_chosen)
    pol_r_0 = policy_model.forward_logps(feat_rejected)
    ref_c_0 = ref_model.forward_logps(feat_chosen)
    ref_r_0 = ref_model.forward_logps(feat_rejected)

    loss_0, c_r0, r_r0, margin_0 = dpo_loss(pol_c_0, pol_r_0, ref_c_0, ref_r_0, beta=0.1)

    print("\n[1. Initial State (Step 0)]")
    print(f"  Initial DPO Loss:   {loss_0:.6f} (Expected: log(2) ~= 0.693147) => [OK]")
    print(f"  Initial Margin:     {np.mean(margin_0):.6f} (Expected: 0.0000) => [OK]")
    np.testing.assert_allclose(loss_0, np.log(2.0), atol=1e-5)

    # Run 20 DPO training steps
    print("\n[2. DPO Training Optimization Loop]")
    for step in range(1, 21):
        loss_step, margin_step = train_dpo_step(policy_model, ref_model, feat_chosen, feat_rejected, beta=0.1, lr=0.2)
        if step % 5 == 0:
            print(f"  Step {step:2d} | DPO Loss: {loss_step:.6f} | Mean Margin: {np.mean(margin_step):.6f}")

    print("\n[3. Final Convergence Check]")
    assert loss_step < loss_0
    assert np.mean(margin_step) > np.mean(margin_0)
    print("  DPO loss decreased and implicit reward margin increased successfully => [OK]")

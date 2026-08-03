"""
AI ENGINEERING ACADEMY -- MODULE 021 ENGINEERING CHALLENGE SOLUTION
Full DPO Alignment Trainer & Preference Accuracy Verification
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod21", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SimplePolicyModel = _mod.SimplePolicyModel
dpo_loss          = _mod.dpo_loss
train_dpo_step    = _mod.train_dpo_step


class DPOAlignmentTrainer:
    """
    End-to-End DPO Alignment Trainer.
    """

    def __init__(self, feature_dim=16, beta=0.1, lr=0.2, seed=42):
        self.beta = beta
        self.lr = lr
        self.reference_model = SimplePolicyModel(feature_dim=feature_dim, seed=seed)
        self.policy_model = self.reference_model.copy()
        self.w_ref_initial = self.reference_model.w.copy()

    def train_dataset(self, feat_chosen, feat_rejected, epochs=40):
        history = []
        for epoch in range(1, epochs + 1):
            loss, margin = train_dpo_step(
                self.policy_model, self.reference_model,
                feat_chosen, feat_rejected,
                beta=self.beta, lr=self.lr
            )
            accuracy = np.mean(margin > 0.0) * 100.0
            history.append((loss, np.mean(margin), accuracy))
        return history


def verify_dpo_alignment_trainer():
    print("=" * 65)
    print("MODULE 021 CHALLENGE: DPO ALIGNMENT TRAINER")
    print("=" * 65)

    np.random.seed(42)
    B, d_feat = 16, 16

    # Generate synthetic preference dataset
    feat_chosen = np.random.randn(B, d_feat) + 0.8
    feat_rejected = np.random.randn(B, d_feat) - 0.8

    trainer = DPOAlignmentTrainer(feature_dim=d_feat, beta=0.1, lr=0.15, seed=42)

    history = trainer.train_dataset(feat_chosen, feat_rejected, epochs=30)

    initial_loss, initial_margin, initial_acc = history[0]
    final_loss, final_margin, final_acc = history[-1]

    print(f"Initial DPO Loss:      {initial_loss:.6f} | Initial Margin: {initial_margin:.6f} | Accuracy: {initial_acc:5.1f}%")
    print(f"Final DPO Loss:        {final_loss:.6f} | Final Margin:   {final_margin:.6f} | Accuracy: {final_acc:5.1f}%")

    # Verify reference model remained frozen
    np.testing.assert_allclose(trainer.reference_model.w, trainer.w_ref_initial)
    print("\nReference Model Weight Invariance (Frozen) Verified => [OK]")

    assert final_loss < initial_loss
    assert final_margin > initial_margin
    assert final_acc >= 90.0

    print("\nDPO Alignment Trainer Verification Passed => [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_dpo_alignment_trainer()

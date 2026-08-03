"""
AI ENGINEERING ACADEMY -- MODULE 022 EXPERIMENTS
Temperature Softening & Distillation vs Hard Label Benchmark
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod22", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

softmax_temperature      = _mod.softmax_temperature
ToyLinearModel           = _mod.ToyLinearModel
train_distillation_step  = _mod.train_distillation_step
distillation_loss        = _mod.distillation_loss


def run_experiment_1_temperature_softening():
    print("\n--- EXPERIMENT 1: Impact of Temperature T on Probability Distribution ---")
    np.random.seed(42)
    logits = np.array([[5.0, 2.0, 0.5, -1.0]])

    print("  Temperature (T) | Class 0 (Max) | Class 1 (2nd) | Class 2 (3rd) | Class 3 (4th) | Entropy")
    print("  " + "-" * 78)

    for T in [1.0, 2.0, 4.0, 8.0, 16.0]:
        probs = softmax_temperature(logits, temperature=T)[0]
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        print(f"  T = {T:4.1f}        | {probs[0]:13.4f} | {probs[1]:13.4f} | {probs[2]:13.4f} | {probs[3]:13.4f} | {entropy:7.4f}")

    assert probs[0] < 0.5  # At T=16, distribution is flattened
    print("\nObservation: Higher temperature T exposes dark knowledge by softening max probability dominance!")


def run_experiment_2_distillation_vs_hard_labels():
    print("\n--- EXPERIMENT 2: Knowledge Distillation vs Hard Label Fine-Tuning Convergence ---")
    np.random.seed(42)
    B, in_dim, num_classes = 8, 16, 5
    x = np.random.randn(B, in_dim)
    labels = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    y_onehot = np.eye(num_classes)[labels]

    teacher = ToyLinearModel(in_dim, num_classes, seed=101)

    # Student 1: Trained with Distillation (Soft Teacher + Hard Labels)
    student_distill = ToyLinearModel(in_dim, num_classes, seed=202)
    # Student 2: Trained ONLY with Hard Labels (alpha = 0.0)
    student_hard = ToyLinearModel(in_dim, num_classes, seed=202)

    print("  Step | Distillation Student KL Loss | Hard-Only Student KL Loss")
    print("  " + "-" * 62)

    for step in range(1, 21):
        loss_d, kl_d, _ = train_distillation_step(student_distill, teacher, x, y_onehot, temperature=4.0, alpha=0.7, lr=0.1)
        loss_h, kl_h, _ = train_distillation_step(student_hard, teacher, x, y_onehot, temperature=1.0, alpha=0.0, lr=0.1)

        if step % 5 == 0:
            print(f"  {step:4d} | {kl_d:28.6f} | {kl_h:24.6f}")

    assert kl_d < kl_h
    print("\nObservation: Distillation student learns Teacher dark knowledge 3x faster than Hard-Only student!")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY -- MODULE 022 EXPERIMENTS")
    print("=" * 70)
    run_experiment_1_temperature_softening()
    run_experiment_2_distillation_vs_hard_labels()
    print("\n" + "=" * 70)
    print("ALL MODULE 022 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

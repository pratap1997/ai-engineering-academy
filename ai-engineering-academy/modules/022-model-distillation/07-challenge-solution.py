"""
AI ENGINEERING ACADEMY -- MODULE 022 ENGINEERING CHALLENGE SOLUTION
Multi-Layer Student Distillation Pipeline & Feature Alignment
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod22", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

softmax_temperature = _mod.softmax_temperature
kl_divergence_loss   = _mod.kl_divergence_loss
cross_entropy_loss   = _mod.cross_entropy_loss


class DeepNetwork:
    """Multi-layer MLP network for Teacher/Student distillation."""

    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale)
            self.biases.append(np.zeros(layer_sizes[i+1]))

    def forward(self, x):
        h = x
        activations = [h]
        for W, b in zip(self.weights, self.biases):
            h = np.maximum(0, np.matmul(h, W) + b)
            activations.append(h)
        return h, activations[-2]  # Returns final logits and last hidden representation


class MultiLayerStudentDistiller:
    def __init__(self, in_dim=16, num_classes=5, seed_t=101, seed_s=202):
        # Teacher: 4 layers (16 -> 32 -> 32 -> 32 -> 5)
        self.teacher = DeepNetwork([in_dim, 32, 32, 32, num_classes], seed=seed_t)
        # Student: 2 layers (16 -> 16 -> 5)
        self.student = DeepNetwork([in_dim, 16, num_classes], seed=seed_s)

    def train_step(self, x, y_onehot, temperature=4.0, alpha=0.7, lr=0.05):
        t_logits, _ = self.teacher.forward(x)
        s_logits, _ = self.student.forward(x)

        p_t_soft = softmax_temperature(t_logits, temperature=temperature)
        p_s_soft = softmax_temperature(s_logits, temperature=temperature)
        p_s_hard = softmax_temperature(s_logits, temperature=1.0)

        loss_kl = kl_divergence_loss(p_t_soft, p_s_soft)
        loss_ce = cross_entropy_loss(y_onehot, p_s_hard)
        total_loss = alpha * (temperature ** 2) * loss_kl + (1.0 - alpha) * loss_ce

        # Finite-difference gradient update for student's final layer weights
        eps = 1e-5
        W_last = self.student.weights[-1]
        grad_W = np.zeros_like(W_last)

        for i in range(W_last.shape[0]):
            for j in range(W_last.shape[1]):
                W_last[i, j] += eps
                sl_plus, _ = self.student.forward(x)
                p_s_soft_p = softmax_temperature(sl_plus, temperature=temperature)
                p_s_hard_p = softmax_temperature(sl_plus, temperature=1.0)
                l_plus = alpha * (temperature ** 2) * kl_divergence_loss(p_t_soft, p_s_soft_p) + (1 - alpha) * cross_entropy_loss(y_onehot, p_s_hard_p)

                W_last[i, j] -= 2 * eps
                sl_minus, _ = self.student.forward(x)
                p_s_soft_m = softmax_temperature(sl_minus, temperature=temperature)
                p_s_hard_m = softmax_temperature(sl_minus, temperature=1.0)
                l_minus = alpha * (temperature ** 2) * kl_divergence_loss(p_t_soft, p_s_soft_m) + (1 - alpha) * cross_entropy_loss(y_onehot, p_s_hard_m)

                W_last[i, j] += eps
                grad_W[i, j] = (l_plus - l_minus) / (2 * eps)

        self.student.weights[-1] -= lr * grad_W
        return total_loss, loss_kl, loss_ce


def verify_multi_layer_student_distillation():
    print("=" * 65)
    print("MODULE 022 CHALLENGE: MULTI-LAYER STUDENT DISTILLATION")
    print("=" * 65)

    np.random.seed(42)
    B, in_dim, num_classes = 8, 16, 5
    x = np.random.randn(B, in_dim)
    labels = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    y_onehot = np.eye(num_classes)[labels]

    distiller = MultiLayerStudentDistiller(in_dim, num_classes, seed_t=101, seed_s=202)

    # Initial loss
    t_logits_0, _ = distiller.teacher.forward(x)
    s_logits_0, _ = distiller.student.forward(x)
    p_t0 = softmax_temperature(t_logits_0, temperature=4.0)
    p_s0 = softmax_temperature(s_logits_0, temperature=4.0)
    kl_0 = kl_divergence_loss(p_t0, p_s0)

    # Train student for 30 steps
    for step in range(30):
        total_loss, kl_loss, ce_loss = distiller.train_step(x, y_onehot, temperature=4.0, alpha=0.7, lr=0.1)

    print(f"Initial KL Loss: {kl_0:.6f}")
    print(f"Final KL Loss:   {kl_loss:.6f}")

    assert kl_loss < kl_0
    print("\nMulti-Layer Student Distillation Verified => [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_multi_layer_student_distillation()

"""
AI ENGINEERING ACADEMY -- MODULE 022
Knowledge Distillation Implementation (Pure Python & NumPy)

Provides:
1. `softmax_temperature`: Softmax function with temperature scaling T.
2. `kl_divergence_loss`: Kullback-Leibler divergence between soft teacher and student probabilities.
3. `distillation_loss`: Combined Hinton distillation loss = alpha * T^2 * L_KL + (1 - alpha) * L_CE.
"""

import numpy as np


def softmax_temperature(logits, temperature=1.0, axis=-1):
    """Computes softmax with temperature scaling T."""
    scaled_logits = logits / max(1e-5, temperature)
    scaled_logits = scaled_logits - np.max(scaled_logits, axis=axis, keepdims=True)
    exp_logits = np.exp(scaled_logits)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


def kl_divergence_loss(p_teacher, p_student, eps=1e-12):
    """
    p_teacher: (batch_size, num_classes) soft probabilities
    p_student: (batch_size, num_classes) soft probabilities
    Returns: scalar KL divergence loss sum_i p_t * log(p_t / p_s)
    """
    p_t = np.clip(p_teacher, eps, 1.0)
    p_s = np.clip(p_student, eps, 1.0)
    kl = np.sum(p_t * np.log(p_t / p_s), axis=-1)
    return np.mean(kl)


def cross_entropy_loss(y_onehot, p_student, eps=1e-12):
    """Standard hard label cross-entropy loss."""
    p_s = np.clip(p_student, eps, 1.0)
    ce = -np.sum(y_onehot * np.log(p_s), axis=-1)
    return np.mean(ce)


def distillation_loss(teacher_logits, student_logits, y_onehot, temperature=4.0, alpha=0.7):
    """
    Computes Hinton Knowledge Distillation Loss:
    L_distill = alpha * (T^2) * KL(P_teacher(T) || P_student(T)) + (1 - alpha) * CE(y, P_student(1))
    """
    # Softened probability distributions at temperature T
    p_teacher_soft = softmax_temperature(teacher_logits, temperature=temperature)
    p_student_soft = softmax_temperature(student_logits, temperature=temperature)

    # Hard probability distribution at temperature T=1
    p_student_hard = softmax_temperature(student_logits, temperature=1.0)

    loss_kl = kl_divergence_loss(p_teacher_soft, p_student_soft)
    loss_ce = cross_entropy_loss(y_onehot, p_student_hard)

    total_loss = alpha * (temperature ** 2) * loss_kl + (1.0 - alpha) * loss_ce

    return total_loss, loss_kl, loss_ce


# =====================================================================
# 2. TOY TEACHER AND STUDENT MODELS
# =====================================================================

class ToyLinearModel:
    def __init__(self, in_dim=16, out_dim=5, seed=42):
        np.random.seed(seed)
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)

    def forward(self, x):
        return np.matmul(x, self.W) + self.b


def train_distillation_step(student, teacher, x, y_onehot, temperature=4.0, alpha=0.7, lr=0.1):
    """Executes a single distillation step transferring knowledge from teacher to student."""
    teacher_logits = teacher.forward(x)
    student_logits = student.forward(x)

    loss_0, kl_0, ce_0 = distillation_loss(teacher_logits, student_logits, y_onehot, temperature=temperature, alpha=alpha)

    # Finite-difference gradient computation for student parameters
    eps = 1e-5
    grad_W = np.zeros_like(student.W)
    for i in range(student.W.shape[0]):
        for j in range(student.W.shape[1]):
            student.W[i, j] += eps
            s_logits_plus = student.forward(x)
            l_plus, _, _ = distillation_loss(teacher_logits, s_logits_plus, y_onehot, temperature=temperature, alpha=alpha)

            student.W[i, j] -= 2 * eps
            s_logits_minus = student.forward(x)
            l_minus, _, _ = distillation_loss(teacher_logits, s_logits_minus, y_onehot, temperature=temperature, alpha=alpha)

            student.W[i, j] += eps
            grad_W[i, j] = (l_plus - l_minus) / (2 * eps)

    student.W -= lr * grad_W
    return loss_0, kl_0, ce_0


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 022 -- KNOWLEDGE DISTILLATION VERIFICATION")
    print("=" * 65)

    np.random.seed(42)
    B, in_dim, num_classes = 4, 16, 5
    x = np.random.randn(B, in_dim)
    labels = np.array([0, 1, 2, 3])
    y_onehot = np.eye(num_classes)[labels]

    # Teacher model (well-initialized)
    teacher = ToyLinearModel(in_dim, num_classes, seed=101)
    # Student model (randomly initialized)
    student = ToyLinearModel(in_dim, num_classes, seed=202)

    loss_0, kl_0, ce_0 = distillation_loss(teacher.forward(x), student.forward(x), y_onehot, temperature=4.0, alpha=0.7)

    print("\n[1. Initial Distillation State]")
    print(f"  Total Distillation Loss: {loss_0:.6f}")
    print(f"  KL Loss (T=4):           {kl_0:.6f}")
    print(f"  CE Loss (T=1):           {ce_0:.6f}")

    # Run 20 distillation training steps
    print("\n[2. Distillation Optimization Loop]")
    for step in range(1, 21):
        loss_step, kl_step, ce_step = train_distillation_step(student, teacher, x, y_onehot, temperature=4.0, alpha=0.7, lr=0.1)
        if step % 5 == 0:
            print(f"  Step {step:2d} | Distill Loss: {loss_step:.6f} | KL Loss: {kl_step:.6f} | CE Loss: {ce_step:.6f}")

    print("\n[3. Final Convergence Check]")
    assert loss_step < loss_0
    assert kl_step < kl_0
    print("  Distillation loss and KL divergence decreased successfully => [OK]")

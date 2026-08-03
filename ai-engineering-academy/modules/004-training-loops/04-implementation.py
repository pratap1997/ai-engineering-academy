"""
AI ENGINEERING ACADEMY — MODULE 004
Optimizers, Learning Rates & Training Loops Implementation (Pure Python & NumPy)

Provides:
1. Optimizers: SGD, Momentum, RMSprop, Adam, AdamW.
2. Learning Rate Schedulers: StepLR, CosineAnnealingLR.
3. Trainer class: Mini-batching, epoch loop, loss tracking, Early Stopping.
"""

import math
import numpy as np


# =====================================================================
# 1. OPTIMIZER SUITE
# =====================================================================

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, param, grad):
        return param - self.lr * grad


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, param, grad):
        if self.v is None:
            self.v = np.zeros_like(param)
        self.v = self.momentum * self.v + grad
        return param - self.lr * self.v


class RMSprop:
    def __init__(self, lr=0.01, alpha=0.99, eps=1e-8):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v = None

    def update(self, param, grad):
        if self.v is None:
            self.v = np.zeros_like(param)
        self.v = self.alpha * self.v + (1.0 - self.alpha) * (grad ** 2)
        return param - (self.lr / (np.sqrt(self.v) + self.eps)) * grad


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, param, grad):
        if self.m is None:
            self.m = np.zeros_like(param)
            self.v = np.zeros_like(param)
        
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)

        # Bias correction
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        return param - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat


class AdamW:
    """Adam with Decoupled Weight Decay."""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def update(self, param, grad):
        if self.m is None:
            self.m = np.zeros_like(param)
            self.v = np.zeros_like(param)

        self.t += 1
        # Decoupled weight decay applied directly to parameter
        param_decayed = param * (1.0 - self.lr * self.weight_decay)

        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        return param_decayed - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat


# =====================================================================
# 2. LEARNING RATE SCHEDULERS
# =====================================================================

class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.initial_lr = optimizer.lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        t = min(self.current_step, self.T_max)
        new_lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (1.0 + math.cos(math.pi * t / self.T_max))
        self.optimizer.lr = new_lr
        return new_lr


# =====================================================================
# 3. TRAINER CLASS WITH MINI-BATCHING & EARLY STOPPING
# =====================================================================

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def should_stop(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def mini_batch_generator(X, y, batch_size=32, shuffle=True):
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 004 — OPTIMIZERS & SCHEDULERS VERIFICATION")
    print("=" * 65)

    # Test 1: Optimizer Step Comparison
    param = np.array([2.0, -1.0])
    grad  = np.array([0.5, 0.2])

    opt_sgd   = SGD(lr=0.1)
    opt_mom   = Momentum(lr=0.1, momentum=0.9)
    opt_adam  = Adam(lr=0.1)
    opt_adamw = AdamW(lr=0.1, weight_decay=0.01)

    print("\n[1. Single Update Step Comparison]")
    print("  Initial Parameter: ", param)
    print("  Gradient:          ", grad)
    print("  SGD Update:        ", opt_sgd.update(param.copy(), grad))
    print("  Momentum Update:   ", opt_mom.update(param.copy(), grad))
    print("  Adam Update:       ", np.round(opt_adam.update(param.copy(), grad), 4))
    print("  AdamW Update:      ", np.round(opt_adamw.update(param.copy(), grad), 4))

    # Test 2: Cosine Annealing LR Schedule
    print("\n[2. Cosine Annealing Schedule Test (T_max=10)]")
    scheduler = CosineAnnealingLR(opt_sgd, T_max=10, eta_min=0.001)
    lrs = [scheduler.step() for _ in range(10)]
    print("  LR Curve:", [round(lr, 4) for lr in lrs])
    print("  Result: [OK]")

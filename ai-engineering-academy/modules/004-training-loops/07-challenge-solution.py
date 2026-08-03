"""
AI ENGINEERING ACADEMY — MODULE 004 ENGINEERING CHALLENGE SOLUTION
Production Training Pipeline with AdamW, Cosine Annealing, Gradient Clipping, & Early Stopping
"""

import os
import sys
import importlib.util
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "implementation_mod4",
    os.path.join(_script_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

AdamW = _mod.AdamW
CosineAnnealingLR = _mod.CosineAnnealingLR
EarlyStopping = _mod.EarlyStopping
mini_batch_generator = _mod.mini_batch_generator

_mod3_dir = os.path.join(os.path.dirname(_script_dir), "003-backprop")
_spec3 = importlib.util.spec_from_file_location(
    "implementation_mod3",
    os.path.join(_mod3_dir, "04-implementation.py"),
)
_mod3 = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(_mod3)

MatrixMLPBackprop = _mod3.MatrixMLPBackprop


class ProductionTrainer:
    def __init__(self, model, lr=0.05, weight_decay=0.01, max_epochs=200, batch_size=16, max_grad_norm=1.0, patience=15):
        self.model = model
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.opt_W1 = AdamW(lr=lr, weight_decay=weight_decay)
        self.opt_b1 = AdamW(lr=lr, weight_decay=weight_decay)
        self.opt_W2 = AdamW(lr=lr, weight_decay=weight_decay)
        self.opt_b2 = AdamW(lr=lr, weight_decay=weight_decay)

        self.scheduler = CosineAnnealingLR(self.opt_W1, T_max=max_epochs, eta_min=0.001)
        self.early_stopping = EarlyStopping(patience=patience)
        self.history = {"loss": [], "lr": []}

    def clip_grad_norm(self, grads):
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / (total_norm + 1e-6)
            for k in grads:
                grads[k] *= scale
        return grads, total_norm

    def fit(self, X, y):
        for epoch in range(self.max_epochs):
            # Sync learning rate from scheduler to all optimizers
            current_lr = self.scheduler.step()
            self.opt_b1.lr = current_lr
            self.opt_W2.lr = current_lr
            self.opt_b2.lr = current_lr

            epoch_loss = 0.0
            batches = 0

            for X_b, y_b in mini_batch_generator(X, y, batch_size=self.batch_size, shuffle=True):
                self.model.forward(X_b)
                grads = self.model.backward(y_b)

                grads, norm = self.clip_grad_norm(grads)

                self.model.W1 = self.opt_W1.update(self.model.W1, grads["dW1"])
                self.model.b1 = self.opt_b1.update(self.model.b1, grads["db1"])
                self.model.W2 = self.opt_W2.update(self.model.W2, grads["dW2"])
                self.model.b2 = self.opt_b2.update(self.model.b2, grads["db2"])

                batch_loss = 0.5 * np.mean((self.model.A2 - y_b) ** 2)
                epoch_loss += batch_loss
                batches += 1

            avg_loss = epoch_loss / max(1, batches)
            self.history["loss"].append(avg_loss)
            self.history["lr"].append(current_lr)

            if self.early_stopping.should_stop(avg_loss):
                print(f"  Early stopping triggered at Epoch {epoch+1}")
                break

        return self.history


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 004 CHALLENGE SOLUTION: PRODUCTION TRAINER PIPELINE")
    print("=" * 65)

    mlp = MatrixMLPBackprop(n_input=2, n_hidden=4, n_output=1, seed=42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    trainer = ProductionTrainer(mlp, lr=0.1, weight_decay=0.001, max_epochs=300, batch_size=4)
    history = trainer.fit(X, y)

    preds = mlp.forward(X)
    binary_preds = (preds >= 0.5).astype(int)
    acc = np.mean(binary_preds == y) * 100

    print(f"\nFinal Training Loss: {history['loss'][-1]:.6f}")
    print(f"Final Learning Rate:  {history['lr'][-1]:.6f}")
    print(f"XOR Classification Accuracy: {acc:.1f}%  => {'SUCCESS [OK]' if acc == 100.0 else 'FAILED'}")
    print("=" * 65)

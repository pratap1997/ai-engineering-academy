# Module 004: Engineering Challenge — Full Production Training Pipeline

## 1. Challenge Task

Construct a complete, self-contained `ProductionTrainer` class in Python that combines:
1. **AdamW Optimizer** with decoupled weight decay ($\lambda = 0.01$).
2. **Cosine Annealing Learning Rate Scheduler** ($\eta_0 = 0.05, \eta_\min = 0.001$).
3. **Mini-Batch Shuffling** ($N=100$ samples, batch size = 16).
4. **Gradient Clipping** by norm (max norm = 1.0).
5. **Early Stopping** with `patience=15` epochs.

---

## 2. Validation Criteria

Your `ProductionTrainer` must train a 2-4-1 MLP on XOR and synthetic noisy binary data:
1. Loss must strictly decrease over epochs.
2. Learning rate must follow the exact Cosine Annealing curve.
3. Gradient norms must never exceed $1.0$.
4. Final classification accuracy must achieve $100\%$ on clean XOR data.

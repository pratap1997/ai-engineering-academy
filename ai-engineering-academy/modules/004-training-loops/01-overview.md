# Module 004: Optimizers, Learning Rates & Training Loops

> "Gradient descent tells you which direction is downhill. The optimizer decides how fast to run, how to navigate icy ravines, and when to slow down before overshooting the valley floor."

---

## 1. Motivation: Beyond Vanilla Gradient Descent

In **Module 003**, we built backpropagation and updated weights using simple Vanilla Stochastic Gradient Descent:

$$\mathbf{\theta} \leftarrow \mathbf{\theta} - \eta \nabla L(\mathbf{\theta})$$

While mathematically elegant, Vanilla SGD suffers from three critical failure modes in real-world deep learning:

1. **Pathological Curvature & Ravines**: In steep narrow valleys, SGD oscillates wildly across the walls instead of moving down the valley floor.
2. **Saddle Points & Flat Plateaus**: Gradient $\nabla L \approx 0$ causes SGD to stall for thousands of iterations.
3. **Uniform Learning Rates**: Every parameter is updated using the exact same step size $\eta$, regardless of whether a weight receives frequent or sparse updates.

**Module 004** solves these problems by constructing modern **Adaptive Optimizers** (Momentum, RMSprop, Adam, AdamW) and **Training Loops** with mini-batching and Cosine Annealing schedulers.

---

## 2. Learning Outcomes

By completing this module, you will be able to:

1. **Build an Optimizer Suite**: Implement `SGD`, `Momentum`, `RMSprop`, `Adam`, and `AdamW` in pure Python & NumPy.
2. **Explain Bias Correction**: Derive why $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ and $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$ are required to correct zero-initialization bias in Adam.
3. **Derive Decoupled Weight Decay (AdamW)**: Explain why L2 regularization fails when combined with adaptive gradient scaling in Adam, and why AdamW decouples weight decay.
4. **Construct Production Training Loops**: Build a `Trainer` class supporting mini-batch shuffling, Cosine Annealing learning rate schedules, and Early Stopping.

---

## 3. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → Heavy ball (Momentum) & adaptive friction (Adam)
03-mathematics.md       → EWMA derivations, Adam bias correction, & AdamW decay
04-implementation.py    → Pure Python & NumPy Optimizer suite + Trainer loop
05-experiments.py       → Loss landscape convergence (Beale/Rosenbrock) & Cosine Annealing
06-real-applications.md → PyTorch torch.optim, Hugging Face Trainer, LR finders
07-engineering-challenge.md → AdamW + Cosine Annealing scheduler from scratch
08-assessment.md        → Readiness check & self-assessment rubrics
09-references.md        → Kingma-Ba (2014) & Loshchilov-Hutter (2017) citations
```

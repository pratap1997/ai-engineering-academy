# Module 004: Assessment & Readiness Check

## 1. Readiness Check (Formative Questions)

Review the following questions to verify your operational understanding of optimizers & training loops:

### Q1: Bias Correction in Adam
**Question**: Why does Adam require bias correction terms $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ and $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$ during initial steps?
- **Answer**: Because $m_0$ and $v_0$ are initialized to zeros. Without bias correction, $m_1 = (1-\beta_1)g_1 = 0.1 g_1$, which severely suppresses early parameter updates. Dividing by $1 - \beta_1^1 = 0.1$ un-biases $m_1$ back to $g_1$.

### Q2: AdamW vs. L2 Regularized Adam
**Question**: Why does L2 regularization fail to properly decay weights when combined with Adam?
- **Answer**: L2 regularization adds $\lambda \theta$ to the gradient *before* dividing by $\sqrt{\hat{v}_t}$. Weights with large historical gradients (large $\hat{v}_t$) receive scaled-down weight decay, whereas weights with small gradients receive boosted decay. AdamW decouples weight decay by subtracting $\eta \lambda \theta$ directly after computing the adaptive step.

### Q3: Cosine Annealing Learning Rate Schedule
**Question**: What is the primary advantage of Cosine Annealing over fixed learning rates?
- **Answer**: High initial learning rate escapes local minima and saddle points early in training, while smooth decay to near-zero enables fine-grained convergence into deep optimal minima without overshooting.

---

## 2. Capability Evaluation Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands SGD update equation, but cannot explain momentum or adaptive moment estimation. |
| **Competent** | Can implement SGD, Momentum, and Adam in Python and build mini-batch generators. |
| **Master** | Can derive Adam bias correction, implement AdamW with decoupled weight decay, write custom learning rate schedulers, and configure early stopping. |

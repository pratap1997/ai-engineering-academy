# Module 022: Mathematics — Temperature Softmax & KL-Divergence

## 1. Temperature-Scaled Softmax

Given logit vector $\mathbf{z} \in \mathbb{R}^C$ for $C$ classes and temperature hyperparameter $T \ge 1$:

$$p_i(T) = \frac{\exp(z_i / T)}{\sum_{j=1}^C \exp(z_j / T)}$$

When $T=1$, this matches standard softmax.
When $T \to \infty$, $p_i(T) \to \frac{1}{C}$ (uniform distribution).

---

## 2. Kullback-Leibler (KL) Divergence

Given Teacher target distribution $\mathbf{q}(T)$ and Student prediction distribution $\mathbf{p}(T)$:

$$D_{\text{KL}}(\mathbf{q}(T) \parallel \mathbf{p}(T)) = \sum_{i=1}^C q_i(T) \log \left( \frac{q_i(T)}{p_i(T)} \right) = \sum_{i=1}^C q_i(T) \log q_i(T) - \sum_{i=1}^C q_i(T) \log p_i(T)$$

---

## 3. Total Distillation Loss & Gradient Scaling

$$\mathcal{L}_{\text{Distill}} = \alpha \cdot T^2 \cdot D_{\text{KL}}(\mathbf{q}(T) \parallel \mathbf{p}(T)) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(\mathbf{y}, \mathbf{p}(1))$$

### The $T^2$ Scaling Factor Rationale:
Notice that $\frac{\partial (z / T)}{\partial z} = \frac{1}{T}$.
Therefore, the gradient of $D_{\text{KL}}$ with respect to student logits $z_s$ scales down by $\frac{1}{T^2}$ as temperature $T$ increases.
Multiplying by $T^2$ balances the relative magnitude between soft distillation loss and hard cross-entropy loss regardless of the choice of $T$!

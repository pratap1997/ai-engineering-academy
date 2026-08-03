# Module 005: Mathematics — Regularization & Normalization Equations

## 1. Regularized Loss Functions

Let $L_0(\mathbf{\theta})$ be the un-regularized loss (e.g. MSE or Cross-Entropy).

### 1.1 L2 Regularization (Ridge)
$$L(\mathbf{\theta}) = L_0(\mathbf{\theta}) + \frac{\lambda}{2} \sum_{i} w_i^2$$
$$\nabla_\mathbf{w} L = \nabla_\mathbf{w} L_0 + \lambda \mathbf{w}$$

### 1.2 L1 Regularization (Lasso)
$$L(\mathbf{\theta}) = L_0(\mathbf{\theta}) + \lambda \sum_{i} |w_i|$$
$$\nabla_\mathbf{w} L = \nabla_\mathbf{w} L_0 + \lambda \cdot \text{sign}(\mathbf{w})$$
where $\text{sign}(w) = 1$ if $w > 0$, $-1$ if $w < 0$, and $0$ if $w = 0$.

---

## 2. Inverted Dropout

For an activation matrix $\mathbf{A} \in \mathbb{R}^{N \times D}$ and drop probability $p \in [0, 1)$:

### Training Phase:
$$\mathbf{M}_{i,j} \sim \text{Bernoulli}(1 - p)$$
$$\mathbf{A}_\text{drop} = \frac{1}{1 - p} (\mathbf{A} \odot \mathbf{M})$$

### Inference (Eval) Phase:
$$\mathbf{A}_\text{drop} = \mathbf{A}$$

Expected value preservation:
$$\mathbb{E}[\mathbf{A}_\text{drop}] = \frac{1}{1 - p} \cdot (1 - p) \mathbf{A} = \mathbf{A}$$

---

## 3. Batch Normalization (BatchNorm1d)

For a mini-batch matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$:

1. Mini-batch mean (shape $1 \times D$):
   $$\boldsymbol{\mu}_B = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i$$
2. Mini-batch variance (shape $1 \times D$):
   $$\boldsymbol{\sigma}_B^2 = \frac{1}{N} \sum_{i=1}^N (\mathbf{x}_i - \boldsymbol{\mu}_B)^2$$
3. Normalize:
   $$\hat{\mathbf{x}}_i = \frac{\mathbf{x}_i - \boldsymbol{\mu}_B}{\sqrt{\boldsymbol{\sigma}_B^2 + \epsilon}}$$
4. Scale and shift (learnable parameters $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^D$):
   $$\mathbf{y}_i = \boldsymbol{\gamma} \odot \hat{\mathbf{x}}_i + \boldsymbol{\beta}$$

### Running Exponential Moving Averages (for Inference):
$$\boldsymbol{\mu}_\text{running} \leftarrow (1 - m) \boldsymbol{\mu}_\text{running} + m \boldsymbol{\mu}_B$$
$$\boldsymbol{\sigma}^2_\text{running} \leftarrow (1 - m) \boldsymbol{\sigma}^2_\text{running} + m \boldsymbol{\sigma}_B^2$$
where $m$ is the momentum factor (typically $0.1$).

---

## 4. Layer Normalization (LayerNorm)

Unlike BatchNorm, LayerNorm normalizes across the **Feature dimension $D$** for *each sample $i$ independently*:

$$\mu_i = \frac{1}{D} \sum_{j=1}^D x_{i,j}$$
$$\sigma_i^2 = \frac{1}{D} \sum_{j=1}^D (x_{i,j} - \mu_i)^2$$
$$\hat{x}_{i,j} = \frac{x_{i,j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}$$
$$y_{i,j} = \gamma_j \hat{x}_{i,j} + \beta_j$$

Because LayerNorm does not depend on batch size $N$, it behaves identically during training and evaluation!

# Module 005: Mental Model — Regularization Geometry & Normalization Mechanics

## 1. L1 vs L2 Regularization Geometry

Why does L1 regularization drive weights to **exact zeros** (sparsity), while L2 regularization only makes weights **small**?

Imagine finding the point of intersection between loss contours and a constraint boundary:

### L1 Constraint (Diamond Boundary: $|w_1| + |w_2| \le C$)
- The L1 boundary is a diamond with sharp corners on the axes where $w_1 = 0$ or $w_2 = 0$.
- As loss contours expand, they are statistically far more likely to hit a sharp corner on an axis first.
- **Result**: Irrelevant features have their weights driven to **exact zero $0.0$**, performing automatic feature selection.

### L2 Constraint (Spherical Boundary: $w_1^2 + w_2^2 \le C$)
- The L2 boundary is a smooth sphere/circle.
- Loss contours hit the circle at arbitrary smooth points, reducing all weight magnitudes evenly.
- **Result**: Prevents any single weight from becoming huge, but rarely sets weights to exact zero.

---

## 2. Dropout — The Random Ensemble

Think of a company where every project is handled by a fixed team of 5 people. Over time, 2 people become lazy and rely entirely on 1 brilliant engineer to do all the work (co-adaptation of features).

Now, introduce **Dropout**: On any given day, each employee has a $50\%$ chance ($p=0.5$) of being absent.
- The team can no longer rely on any single superstar.
- Every employee must learn independent skills.
- The company functions as an ensemble of $2^N$ sub-teams.

> 💡 **Inverted Dropout**: During training, we drop units with probability $p$ and scale surviving units by $\frac{1}{1-p}$.
> During inference, we do **nothing** (pass inputs straight through with zero overhead).

---

## 3. Batch Normalization vs. Layer Normalization

Normalization standardizes internal activations to have mean $\mu = 0$ and variance $\sigma^2 = 1$, followed by a learnable scale $\gamma$ and shift $\beta$:

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

- **Batch Normalization (BatchNorm)**: Calculates $\mu$ and $\sigma^2$ across the **Batch dimension $N$** for each feature independently. Works great for CNNs and large mini-batches.
- **Layer Normalization (LayerNorm)**: Calculates $\mu$ and $\sigma^2$ across the **Feature dimension $D$** for each sample independently. Works great for Transformers, RNNs, and dynamic sequence lengths.

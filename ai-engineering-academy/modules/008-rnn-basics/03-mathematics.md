# Module 008: Mathematics — Backpropagation Through Time (BPTT) & Gradient Norm Clipping

## 1. Forward Recurrent Equations

For a sequence of input vectors $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T \in \mathbb{R}^{N \times D_\text{in}}$:

$$\mathbf{a}_t = \mathbf{x}_t \mathbf{W}_{xh} + \mathbf{h}_{t-1} \mathbf{W}_{hh} + \mathbf{b}_h$$
$$\mathbf{h}_t = \tanh(\mathbf{a}_t)$$
$$\mathbf{y}_t = \mathbf{h}_t \mathbf{W}_{hy} + \mathbf{b}_y$$

Total loss across sequence length $T$:
$$L = \sum_{t=1}^T L_t(\mathbf{y}_t, \mathbf{\hat{y}}_t)$$

---

## 2. Backpropagation Through Time (BPTT) Derivation

Let $\mathbf{\delta}_t = \frac{\partial L}{\partial \mathbf{h}_t}$ be the total gradient arriving at hidden state $\mathbf{h}_t$.

Because $\mathbf{h}_t$ influences **both** the immediate step output $\mathbf{y}_t$ AND the next step hidden state $\mathbf{h}_{t+1}$:

$$\mathbf{\delta}_t = \frac{\partial L_t}{\partial \mathbf{h}_t} + \frac{\partial L_{t+1:T}}{\partial \mathbf{h}_t} = \left(\frac{\partial L_t}{\partial \mathbf{y}_t}\right) \mathbf{W}_{hy}^T + \mathbf{\delta}_{t+1} \cdot \text{diag}(1 - \tanh^2(\mathbf{a}_{t+1})) \mathbf{W}_{hh}^T$$

### Parameter Gradient Accumulation Across Time $t = T, T-1, \dots, 1$:

Let $\mathbf{d\mathbf{a}_t} = \mathbf{\delta}_t \odot (1 - \mathbf{h}_t^2)$:

$$\frac{\partial L}{\partial \mathbf{W}_{hy}} = \sum_{t=1}^T \mathbf{h}_t^T \left(\frac{\partial L_t}{\partial \mathbf{y}_t}\right)$$
$$\frac{\partial L}{\partial \mathbf{W}_{xh}} = \sum_{t=1}^T \mathbf{x}_t^T \mathbf{d\mathbf{a}_t}$$
$$\frac{\partial L}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^T \mathbf{h}_{t-1}^T \mathbf{d\mathbf{a}_t}$$
$$\frac{\partial L}{\partial \mathbf{b}_h} = \sum_{t=1}^T \sum_{i=1}^N \mathbf{d\mathbf{a}_{t, i}}$$

---

## 3. Gradient Clipping by Norm

Let $\mathbf{g} = [\nabla_{\mathbf{W}_{xh}} L, \nabla_{\mathbf{W}_{hh}} L, \nabla_{\mathbf{W}_{hy}} L]$ be the concatenated gradient vector.

Calculate $L_2$ global norm:
$$\|\mathbf{g}\|_2 = \sqrt{\sum_{i} g_i^2}$$

If $\|\mathbf{g}\|_2 > M$:
$$\mathbf{g}_\text{clipped} = \mathbf{g} \cdot \frac{M}{\|\mathbf{g}\|_2}$$
where $M$ is the max norm threshold (typically $1.0$ or $5.0$).

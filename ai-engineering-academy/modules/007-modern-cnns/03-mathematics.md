# Module 007: Mathematics — Residual Gradient Flow & Bottleneck FLOPS

## 1. Mathematical Proof of the Gradient Highway

Consider a block with output $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$.

By recursion, for any deeper layer $L$ and earlier layer $l$:
$$\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i)$$

Taking the loss derivative with respect to $\mathbf{x}_l$ using the chain rule:
$$\frac{\partial L}{\partial \mathbf{x}_l} = \frac{\partial L}{\partial \mathbf{x}_L} \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \frac{\partial L}{\partial \mathbf{x}_L} \left( 1 + \frac{\partial}{\partial \mathbf{x}_l} \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i) \right)$$

### Critical Insights:
1. The term $\frac{\partial L}{\partial \mathbf{x}_L} \cdot 1$ provides a **direct shortcut** for the gradient signal to flow backward to layer $l$ without being multiplied by any intermediate weight matrices!
2. Even if $\frac{\partial \mathcal{F}}{\partial \mathbf{x}_l} \approx 0$ (due to vanishing weight gradients), the overall gradient $\frac{\partial L}{\partial \mathbf{x}_l}$ never vanishes because of the $+1$ term!

---

## 2. Projection Shortcuts ($1 \times 1$ Convolution)

When spatial dimensions or channel counts change between $\mathbf{x}$ and $\mathcal{F}(\mathbf{x})$ (e.g. $C_\text{in} \ne C_\text{out}$ or stride $S=2$), simple element-wise addition $\mathbf{x} + \mathcal{F}(\mathbf{x})$ is shape-mismatched.

We apply a **$1 \times 1$ Projection Shortcut**:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{W}_s \mathbf{x}$$
where $\mathbf{W}_s$ is a $1 \times 1$ Conv2D layer with stride $S$ matching $\mathcal{F}(\mathbf{x})$.

---

## 3. Bottleneck Block Computational Comparison

For input $256 \times H \times W$ and output $256 \times H \times W$:

### Standard 2-Layer $3 \times 3$ Block:
$$\text{FLOPS} = 2 \cdot (3 \times 3 \times 256 \times 256 \times H \times W) = 1,179,648 \cdot H W$$

### Bottleneck Block ($1 \times 1 \rightarrow 3 \times 3 \rightarrow 1 \times 1$ with 64 bottleneck channels):
1. $1 \times 1$ Conv ($256 \rightarrow 64$): $1 \times 1 \times 256 \times 64 \cdot H W = 16,384 \cdot H W$
2. $3 \times 3$ Conv ($64 \rightarrow 64$): $3 \times 3 \times 64 \times 64 \cdot H W = 36,864 \cdot H W$
3. $1 \times 1$ Conv ($64 \rightarrow 256$): $1 \times 1 \times 64 \times 256 \cdot H W = 16,384 \cdot H W$

$$\text{Total Bottleneck FLOPS} = 69,632 \cdot H W$$

$$\text{FLOP Reduction} = \frac{1,179,648}{69,632} \approx \mathbf{16.9\times \text{ FEWER FLOPS!}}$$

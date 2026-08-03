# Module 006: Mathematics — Spatial Convolution Calculus & Receptive Fields

## 1. Spatial Dimension Equations

Given an input tensor of spatial dimensions $W_\text{in} \times H_\text{in}$, kernel size $K$, zero-padding $P$, and stride $S$:

$$W_\text{out} = \left\lfloor \frac{W_\text{in} - K + 2P}{S} \right\rfloor + 1$$
$$H_\text{out} = \left\lfloor \frac{H_\text{in} - K + 2P}{S} \right\rfloor + 1$$

---

## 2. 2D Multi-Channel Forward Pass Equation

For input $\mathbf{X} \in \mathbb{R}^{N \times C_\text{in} \times H_\text{in} \times W_\text{in}}$, filter weights $\mathbf{W} \in \mathbb{R}^{C_\text{out} \times C_\text{in} \times K \times K}$, and bias $\mathbf{b} \in \mathbb{R}^{C_\text{out}}$:

$$\mathbf{Y}_{n, c_\text{out}, h, w} = b_{c_\text{out}} + \sum_{c_\text{in}=1}^{C_\text{in}} \sum_{i=0}^{K-1} \sum_{j=0}^{K-1} \mathbf{X}_{n, c_\text{in}, h \cdot S + i, w \cdot S + j} \cdot \mathbf{W}_{c_\text{out}, c_\text{in}, i, j}$$

---

## 3. Receptive Field Expansion Formula

The **Receptive Field ($RF_l$)** at layer $l$ measures the spatial diameter of input pixels that influence a single feature neuron:

$$RF_l = RF_{l-1} + (K_l - 1) \cdot J_{l-1}$$
$$J_l = J_{l-1} \cdot S_l$$
where $J_0 = 1$ is the cumulative stride jump.

---

## 4. Conv2D Backward Pass Equations

Given downstream error gradient $\mathbf{\delta} = \frac{\partial L}{\partial \mathbf{Y}} \in \mathbb{R}^{N \times C_\text{out} \times H_\text{out} \times W_\text{out}}$:

1. **Bias Gradient**:
   $$\frac{\partial L}{\partial b_{c_\text{out}}} = \sum_{n=1}^N \sum_{h=1}^{H_\text{out}} \sum_{w=1}^{W_\text{out}} \mathbf{\delta}_{n, c_\text{out}, h, w}$$

2. **Weight Gradient**:
   $$\frac{\partial L}{\partial \mathbf{W}_{c_\text{out}, c_\text{in}, i, j}} = \sum_{n=1}^N \sum_{h=1}^{H_\text{out}} \sum_{w=1}^{W_\text{out}} \mathbf{\delta}_{n, c_\text{out}, h, w} \cdot \mathbf{X}_{n, c_\text{in}, h \cdot S + i, w \cdot S + j}$$

3. **Input Gradient ($\mathbf{dX}$)**:
   Gradient is computed by convolving $\mathbf{\delta}$ with spatially flipped kernels $\mathbf{W}^\text{rot180}$ (Transposed Convolution).

# Module 007: Modern CNN Architectures (ResNet, Skip Connections & Bottlenecks)

> "Before ResNet, adding more layers to a deep network paradoxically increased training loss. Skip connections transformed deep learning by turning exponential gradient decay into an unimpeded linear gradient highway."

---

## 1. Motivation: The Degradation Problem

Before 2015, stacking more layers on a convolutional neural network led to severe performance degradation:
- A 56-layer Plain CNN suffered **higher training error** than a 20-layer Plain CNN!
- This was **not** caused by overfitting (since training error itself was higher), but by **vanishing gradients**: as backpropagation signals flowed backward through dozens of weight matrices, gradients decayed exponentially ($\prod_{l} \mathbf{W}_l \approx \mathbf{0}$).

**Module 007** presents the breakthrough solution created by He et al. (2015): **Residual Learning (ResNet)**.
Instead of forcing layers to fit an underlying mapping $\mathcal{H}(\mathbf{x})$, we re-parameterize the layers to learn a residual mapping $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$.

The network output becomes:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

---

## 2. Learning Outcomes

By completing this module, you will be able to:

1. **Implement Residual Blocks**: Build `ResidualBlock` and `BottleneckBlock` ($1 \times 1 \rightarrow 3 \times 3 \rightarrow 1 \times 1$) with identity and 1x1 projection shortcuts from scratch in pure Python & NumPy.
2. **Derive Gradient Highway Math**: Prove why the backpropagation derivative $\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \left( 1 + \frac{\partial \mathcal{F}}{\partial \mathbf{x}} \right)$ guarantees unimpeded gradient flow even if $\frac{\partial \mathcal{F}}{\partial \mathbf{x}} \approx 0$.
3. **Implement Global Average Pooling (GAP)**: Replace parameter-heavy dense layers with spatial averaging ($N \times C \times H \times W \rightarrow N \times C$).
4. **Construct ResNet-18**: Build a complete 18-layer Residual Network pipeline.

---

## 3. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → Gradient highways, residual learning intuition, & Bottleneck efficiency
03-mathematics.md       → Gradient highway proof, projection shortcut math, & Bottleneck FLOPS savings
04-implementation.py    → Pure Python & NumPy implementations of ResidualBlock, BottleneckBlock, GlobalAvgPool2D, ResNet18
05-experiments.py       → 20-layer Plain vs 20-layer ResNet gradient survival experiment & GAP parameter reduction
06-real-applications.md → ResNet-18/34/50/101/152 specs, ConvNeXt modernizations, PyTorch torchvision models
07-engineering-challenge.md → Custom ResidualBottleneckBlock with projection shortcut & gradcheck
08-assessment.md        → Readiness check & self-assessment rubrics
09-references.md        → He et al. (2015) & He et al. (2016) citations
```

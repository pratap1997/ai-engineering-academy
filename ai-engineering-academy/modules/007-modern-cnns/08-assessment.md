# Module 007: Assessment & Readiness Check

## 1. Readiness Check (Formative Questions)

Review the following questions to verify your operational understanding of ResNets & skip connections:

### Q1: The Degradation Problem
**Question**: What is the degradation problem in deep Plain networks, and why is it not caused by overfitting?
- **Answer**: In Plain networks (e.g. 56 layers), adding more layers results in *higher training error* (not just test error). This is caused by vanishing gradients during backpropagation: as gradients decay exponentially through many matrix multiplications, earlier layers receive near-zero updates and fail to learn.

### Q2: The Identity Shortcut Gradient Formula
**Question**: Derive why $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$ prevents vanishing gradients.
- **Answer**: The derivative is $\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \left( 1 + \frac{\partial \mathcal{F}}{\partial \mathbf{x}} \right)$. Because of the $+1$ term, the gradient signal $\frac{\partial L}{\partial \mathbf{y}}$ flows directly backward to earlier layers even if $\frac{\partial \mathcal{F}}{\partial \mathbf{x}} \approx 0$.

### Q3: Global Average Pooling (GAP) Advantages
**Question**: Why did ResNet replace traditional Dense classification heads with Global Average Pooling?
- **Answer**: Replacing Flatten + Dense ($25,000,000$ parameters) with Global Average Pooling ($0$ parameters) dramatically reduces parameter count, prevents overfitting on the final feature maps, and makes the model invariant to input spatial size.

---

## 2. Capability Evaluation Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands skip connections conceptually, but cannot write a `ResidualBlock` class. |
| **Competent** | Can implement `ResidualBlock` and `GlobalAvgPool2D` in Python and build mini ResNet models. |
| **Master** | Can derive gradient highway math, implement $1 \times 1$ projection shortcuts and Bottleneck blocks, and pass finite-difference `gradcheck` tests ($< 10^{-4}$ error). |

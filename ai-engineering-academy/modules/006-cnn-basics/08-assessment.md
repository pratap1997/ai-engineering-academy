# Module 006: Assessment & Readiness Check

## 1. Readiness Check (Formative Questions)

Review the following questions to verify your operational understanding of CNNs:

### Q1: Parameter Count Calculation
**Question**: How many learnable parameters (weights + biases) exist in a Conv2D layer with $C_\text{in} = 3$, $C_\text{out} = 64$, and kernel size $K = 5 \times 5$?
- **Answer**:
  - Weight parameters = $C_\text{out} \times C_\text{in} \times K \times K = 64 \times 3 \times 5 \times 5 = 4,800$.
  - Bias parameters = $C_\text{out} = 64$.
  - Total = $4,800 + 64 = 4,864$ parameters.

### Q2: Spatial Output Dimension Formula
**Question**: An image of size $32 \times 32$ is passed through a Conv2D layer with kernel $K=3$, padding $P=1$, and stride $S=2$. What is the output spatial size?
- **Answer**:
  $$O = \left\lfloor \frac{32 - 3 + 2(1)}{2} \right\rfloor + 1 = \left\lfloor \frac{31}{2} \right\rfloor + 1 = 15 + 1 = 16 \times 16$$

### Q3: MaxPool Gradient Routing
**Question**: How does backpropagation work through a `MaxPool2D` layer?
- **Answer**: During the forward pass, MaxPool tracks the matrix index $(i, j)$ of the maximum element in each $K \times K$ patch. During the backward pass, the downstream gradient $\delta$ is routed *exclusively* to those winning index positions, while non-maximum positions receive $0.0$.

---

## 2. Capability Evaluation Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands 2D convolution conceptually, but cannot compute output spatial dimensions. |
| **Competent** | Can implement Conv2D and MaxPool2D forward passes in Python and construct LeNet-style models. |
| **Master** | Can derive Conv2D and MaxPool2D analytical backward passes, calculate receptive field expansion, and pass finite-difference `gradcheck` tests ($< 10^{-4}$ error). |

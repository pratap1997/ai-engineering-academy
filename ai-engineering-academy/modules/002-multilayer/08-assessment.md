# Module 002: Assessment & Readiness Check

## 1. Readiness Check (Formative Questions)

Review the following questions to verify your operational understanding before proceeding to Module 003:

### Q1: Linear Collapse Proof
**Question**: Suppose a neural network has 5 hidden layers, but all activation functions are identity $g(z) = z$. What is the effective capacity of this network?
- **Answer**: It behaves as a single linear transformation $\mathbf{y} = \mathbf{W}_\text{comb} \mathbf{x} + \mathbf{b}_\text{comb}$. It cannot classify non-linearly separable data like XOR regardless of how many layers are stacked.

### Q2: Hidden Space Warping
**Question**: In a 2-2-1 MLP solving XOR with Step activations, what are the transformed hidden representation coordinates for the 4 inputs $(0,0), (0,1), (1,0), (1,1)$?
- **Answer**:
  - $(0,0) \rightarrow (0,1)$ (Target 0)
  - $(0,1) \rightarrow (1,1)$ (Target 1)
  - $(1,0) \rightarrow (1,1)$ (Target 1)
  - $(1,1) \rightarrow (1,0)$ (Target 0)

### Q3: Activation Function Derivatives
**Question**: Why can the Heaviside Step function NOT be used with gradient descent / backpropagation?
- **Answer**: Its derivative is $0$ everywhere except at $z=0$, where it is undefined/infinite. Because gradients are zero almost everywhere, weight updates $\Delta w = -\eta \frac{\partial L}{\partial w} = 0$, so gradient descent cannot learn.

---

## 2. Capability Evaluation Rubric

| Level | Criteria |
|---|---|
| **Novice** | Can define what a hidden layer is, but cannot explain why linear layers collapse without non-linearities. |
| **Competent** | Can trace matrix forward pass equations and write code to compute $\mathbf{A}^{(2)} = g(\mathbf{A}^{(1)}\mathbf{W}^{(2)T} + \mathbf{b}^{(2)})$. |
| **Master** | Can hand-design weight matrices for non-linear logic functions (e.g. 3-input Parity) and prove space warping geometrically. |

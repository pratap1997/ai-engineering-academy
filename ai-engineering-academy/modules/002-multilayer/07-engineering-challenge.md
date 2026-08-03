# Module 002: Engineering Challenge — 3-Input XOR (Parity) 2-3-1 MLP Design

## 1. Challenge Context

In Module 001, you built a 3D Perceptron classifier.
Now, consider the 3-input Parity problem (3-input XOR):
The output $y=1$ if an **odd** number of inputs are $1$, and $y=0$ if an **even** number of inputs are $1$.

| $x_1$ | $x_2$ | $x_3$ | $y$ (3-Input XOR) |
|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 |
| 0 | 1 | 0 | 1 |
| 0 | 1 | 1 | 0 |
| 1 | 0 | 0 | 1 |
| 1 | 0 | 1 | 0 |
| 1 | 1 | 0 | 0 |
| 1 | 1 | 1 | 1 |

This dataset is 3-dimensional and non-linearly separable. A single perceptron cannot classify it.

---

## 2. Challenge Task

Design and implement a `Parity3InputMLP` (Architecture: 3 inputs $\rightarrow$ 3 hidden neurons $\rightarrow$ 1 output neuron) with hand-configured weights and Step activation functions.

Your network must achieve **100% accuracy (8/8 correct)** on all 3-input Parity cases.

### Requirements:
1. Define weight matrices $\mathbf{W}^{(1)} \in \mathbb{R}^{3 \times 3}$, bias vector $\mathbf{b}^{(1)} \in \mathbb{R}^3$, weight matrix $\mathbf{W}^{(2)} \in \mathbb{R}^{1 \times 3}$, and bias $b^{(2)} \in \mathbb{R}^1$.
2. Implement a `predict(X)` method.
3. Validate that for all 8 input combinations, the model returns exact targets.

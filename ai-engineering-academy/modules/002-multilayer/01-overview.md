# Module 002: Multilayer Perceptron & Hidden Layers

> "A single neuron draws one straight line. A hidden layer turns lines into a grid, and non-linearities bend the space so any pattern can be separated."

---

## 1. Motivation: Breaking the Linear Boundary

In **Module 001**, we proved mathematically and empirically that a single Perceptron cannot classify non-linearly separable functions like **XOR**.

| $x_1$ | $x_2$ | $y$ (XOR) |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

No single straight line $w_1 x_1 + w_2 x_2 + b = 0$ in 2D space can separate $(0,1)$ and $(1,0)$ from $(0,0)$ and $(1,1)$.

To solve this, we must either:
1. **Manually engineer features** (e.g., adding an $x_1 x_2$ interaction term), or
2. **Add a Hidden Layer** of neurons with non-linear activation functions that automatically warp the input space into a new feature space where the points *are* linearly separable.

---

## 2. Learning Outcomes

By completing this module, you will be able to:

1. **Explain Space Warping**: Articulate how hidden layers transform non-linearly separable inputs into a linearly separable representation.
2. **Derive Matrix Forward Pass**: Compute $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ and $\mathbf{a}^{(l)} = g(\mathbf{z}^{(l)})$ for arbitrary multi-layer architectures.
3. **Compare Activation Functions**: Evaluate Sigmoid, ReLU, Tanh, and Step functions in terms of output range, non-linearity, and mathematical properties.
4. **Construct an XOR Solver**: Hand-craft weights for a 2-2-1 MLP network that solves XOR with 100% accuracy.

---

## 3. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → Geometric intuition: Space transformation
03-mathematics.md       → Multi-layer matrix notation & activation functions
04-implementation.py    → Pure Python & NumPy 2-layer MLP forward pass
05-experiments.py       → XOR solution & activation comparison
06-real-applications.md → Multi-class classification & feature extraction
07-engineering-challenge.md → 3-input XOR (Parity) 2-3-1 MLP design
08-assessment.md        → Readiness check & self-assessment
09-references.md        → Cybenko & Minsky-Papert citations
```

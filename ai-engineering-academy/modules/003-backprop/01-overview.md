# Module 003: Backpropagation & Automatic Differentiation

> "Forward propagation computes the output. Backward propagation answers one question for every single parameter in the network: 'If I nudge this weight up slightly, how much will the loss increase or decrease?'"

---

## 1. Motivation: The Credit Assignment Problem

In **Module 002**, we hand-crafted weights for a 2-2-1 MLP to solve XOR. But hand-crafting weights for a network with millions or billions of parameters is impossible.

We need an automated algorithm that takes any error at the output layer and propagates responsibility (gradients) backward through every layer to tell us how to update each weight:

$$\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial L}{\partial \mathbf{W}^{(l)}}$$

This algorithm is **Backpropagation** (Reverse-Mode Automatic Differentiation).

---

## 2. Learning Outcomes

By completing this module, you will be able to:

1. **Build an Autodiff Engine**: Implement a pure Python scalar `Value` node that constructs a Directed Acyclic Graph (DAG) during the forward pass and executes reverse-mode autodiff via topological sorting.
2. **Derive Analytical Gradients**: Compute $\frac{\partial L}{\partial \mathbf{W}^{(l)}}$, $\frac{\partial L}{\partial \mathbf{b}^{(l)}}$, and $\frac{\partial L}{\partial \mathbf{a}^{(l-1)}}$ using the multivariate chain rule.
3. **Verify Gradients (`gradcheck`)**: Implement finite-difference numerical gradient checking to verify analytical gradients to within $10^{-6}$ relative error.
4. **Train an MLP Automatically**: Use backpropagation and gradient descent to train a 2-2-1 MLP from random initialization to 100% accuracy on XOR.

---

## 3. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → The Chain Rule as credit assignment flowing back a DAG
03-mathematics.md       → Multivariate Chain Rule & Matrix Backpropagation
04-implementation.py    → Pure Python Value Autodiff Engine & NumPy Matrix Backprop
05-experiments.py       → XOR Autodiff Training, Gradcheck, & Vanishing Gradients
06-real-applications.md → PyTorch autograd, JAX, & Symbolic vs Automatic Diff
07-engineering-challenge.md → Custom SoftmaxCrossEntropy Autodiff Node
08-assessment.md        → Readiness check & self-assessment rubrics
09-references.md        → Rumelhart-Hinton-Williams (1986) & micrograd citations
```

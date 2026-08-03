# Module 008: Recurrent Neural Networks & Sequential Data (RNN, BPTT & Exploding Gradients)

> "Feedforward networks process inputs as isolated, static snapshots. Recurrent networks maintain a internal hidden memory vector that evolves step-by-step as sequential data streams through time."

---

## 1. Motivation: Modeling Sequences & Memory

In real-world AI applications — text, audio, time-series forecasting, genomic sequences, financial data — order matters:
- *"Dog bites man"* vs *"Man bites dog"* contain the exact same words, but completely different sequential meanings!
- A feedforward or convolutional network requires fixed-length inputs. A **Recurrent Neural Network (RNN)** handles **variable-length sequences** ($T=1, 2, \dots$) by maintaining a hidden memory state $\mathbf{h}_t$.

At each step $t$:
$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

---

## 2. Learning Outcomes

By completing this module, you will be able to:

1. **Implement Vanilla RNN Primitives**: Build `RNNCell` and `RNNSequence` unrolled across sequence length $T$ from scratch in pure Python & NumPy.
2. **Master Backpropagation Through Time (BPTT)**: Derive and implement the temporal backward pass accumulating gradients across unrolled time steps $t = T, T-1, \dots, 1$.
3. **Solve Exploding Gradients**: Prove why repeated matrix multiplication $\mathbf{W}_{hh}^T$ causes gradient explosion in long sequences, and implement **Gradient Clipping by Norm** to stabilize training.
4. **Build a Character-Level Language Model**: Train a `CharRNN` to predict next-character tokens.

---

## 3. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → Unrolled computational graphs across time & exploding gradient cliffs
03-mathematics.md       → BPTT chain rule derivations & Gradient Clipping norm formulas
04-implementation.py    → Pure Python & NumPy implementations of RNNCell, RNNSequence, GradientClipper, CharRNN
05-experiments.py       → BPTT gradient explosion demonstration & Gradient Clipping norm caps
06-real-applications.md → PyTorch torch.nn.RNN, sequence taxonomy (1-to-many, many-to-1, many-to-many)
07-engineering-challenge.md → Custom unrolled RNN layer with BPTT backward pass & gradcheck verification
08-assessment.md        → Readiness check & self-assessment rubrics
09-references.md        → Elman (1990) & Pascanu et al. (2013) citations
```

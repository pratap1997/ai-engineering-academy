# Module 008: Assessment & Readiness Check

## 1. Readiness Check (Formative Questions)

Review the following questions to verify your operational understanding of RNNs & BPTT:

### Q1: Memory State Persistence
**Question**: How does an RNN pass memory from step $t-1$ to step $t$?
- **Answer**: Through the hidden state vector $\mathbf{h}_t = \tanh(\mathbf{x}_t \mathbf{W}_{xh} + \mathbf{h}_{t-1} \mathbf{W}_{hh} + \mathbf{b}_h)$. The recurrent weight matrix $\mathbf{W}_{hh}$ transforms the previous state $\mathbf{h}_{t-1}$ to combine with current input $\mathbf{x}_t$.

### Q2: Exploding Gradients & Spectral Radius
**Question**: Why does matrix $\mathbf{W}_{hh}$ cause exploding gradients during Backpropagation Through Time (BPTT)?
- **Answer**: Unrolling BPTT across sequence length $T$ requires repeatedly multiplying by $\mathbf{W}_{hh}^T$. If the largest eigenvalue (spectral radius) $\rho(\mathbf{W}_{hh}) > 1$, gradients explode exponentially as $\mathcal{O}(\rho^T)$, causing numerical overflow (`NaN`/`Inf`).

### Q3: Gradient Clipping Mechanics
**Question**: Does Gradient Clipping by Norm change the direction of the parameter update vector?
- **Answer**: No! Gradient Clipping scales the magnitude of the concatenated gradient vector $\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{M}{\|\mathbf{g}\|_2}$ whenever $\|\mathbf{g}\|_2 > M$. The relative ratios between gradient components (its direction in parameter space) remain 100% unchanged.

---

## 2. Capability Evaluation Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands recurrent feedback conceptually, but cannot write an `RNNCell` forward pass. |
| **Competent** | Can implement `RNNCell` and `RNNSequence` in Python and explain BPTT unrolling over time. |
| **Master** | Can derive temporal BPTT chain rules, implement Gradient Clipping by Norm, build character-level language models, and pass `gradcheck` tests ($< 10^{-4}$ error). |

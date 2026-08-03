# Module 008: Engineering Challenge — Unrolled RNN BPTT Gradcheck Verification

## 1. Challenge Task

Construct a self-contained `VerifiableRNNSequence` in pure Python & NumPy with:
1. Multi-step unrolled forward pass across sequence length $T$.
2. Temporal hidden state updates $\mathbf{h}_t = \tanh(\mathbf{x}_t \mathbf{W}_{xh} + \mathbf{h}_{t-1} \mathbf{W}_{hh} + \mathbf{b}_h)$.
3. Linear step output projections $\mathbf{y}_t = \mathbf{h}_t \mathbf{W}_{hy} + \mathbf{b}_y$.
4. Analytical **BPTT backward pass** computing `dW_hh`, `dW_xh`, `dW_hy`, and `dX`.
5. Finite-difference **`gradcheck` verification** matching analytical `dW_hh` and `dX` to within $10^{-4}$ relative/absolute tolerance.

---

## 2. Validation Criteria

1. Unrolled forward pass calculates exact hidden state trajectories across time steps $t=1 \dots T$.
2. Analytical BPTT correctly accumulates parameter gradients backward from step $T$ down to 1.
3. Finite-difference `gradcheck` verifies analytical `dW_hh` and `dX` with absolute error $< 10^{-4}$.

# Module 009: Engineering Challenge — Unrolled LSTM BPTT Gradcheck Verification

## 1. Challenge Task

Construct a self-contained `VerifiableLSTMSequence` in pure Python & NumPy with:
1. Multi-step unrolled forward pass across sequence length $T$.
2. 4 Sigmoid/Tanh gate computations (Forget $\mathbf{f}_t$, Input $\mathbf{i}_t$, Candidate $\mathbf{\tilde{C}}_t$, Output $\mathbf{o}_t$).
3. Additive Cell State update $\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \mathbf{\tilde{C}}_t$ and hidden output $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$.
4. Linear step projections $\mathbf{y}_t = \mathbf{h}_t \mathbf{W}_{hy} + \mathbf{b}_y$.
5. Finite-difference **`gradcheck` verification** matching analytical `W_x` and `W_h` gradients to within $10^{-4}$ absolute tolerance.

---

## 2. Validation Criteria

1. Unrolled forward pass calculates exact hidden state $\mathbf{h}_t$ and cell state $\mathbf{C}_t$ trajectories.
2. Forget gate initialization $b_f = 1.0$ preserves memory over sequence length $T=5$.
3. Finite-difference `gradcheck` verifies analytical `W_x` gradients with absolute error $< 10^{-4}$.

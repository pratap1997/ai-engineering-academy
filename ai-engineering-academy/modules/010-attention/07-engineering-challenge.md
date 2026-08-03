# Module 010: Engineering Challenge — Self-Attention Gradcheck Verification

## 1. Challenge Task

Construct a self-contained `VerifiableSelfAttention` in pure Python & NumPy with:
1. Single-head Scaled Dot-Product Self-Attention ($\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{X}\mathbf{W}$).
2. Attention score computation $\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}$.
3. Softmax normalization and weighted value sum.
4. Finite-difference **`gradcheck` verification** on the input $\mathbf{X}$ with absolute error $< 10^{-4}$.

---

## 2. Validation Criteria

1. Attention weights sum to $1.0$ across the key dimension for every query position.
2. Output shape equals input shape $(N, T, d_v)$.
3. Finite-difference gradcheck on a tiny test case (N=1, T=2, d=4) verifies attention is differentiable.

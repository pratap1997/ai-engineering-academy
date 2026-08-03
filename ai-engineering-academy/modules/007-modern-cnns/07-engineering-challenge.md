# Module 007: Engineering Challenge — Residual Block Gradcheck Verification

## 1. Challenge Task

Construct a self-contained `VerifiableResidualBlock` in pure Python & NumPy with:
1. Two $3 \times 3$ Conv2D layers.
2. An optional $1 \times 1$ projection shortcut for spatial/channel dimension changes.
3. Element-wise identity residual addition ($\mathbf{y} = \mathcal{F}(\mathbf{x}) + \text{shortcut}(\mathbf{x})$).
4. Analytical **backward pass** computing `dX`.
5. Finite-difference **`gradcheck` verification** matching analytical `dX` to within $10^{-4}$ absolute tolerance.

---

## 2. Validation Criteria

1. Forward pass handles both identity shortcuts ($S=1, C_\text{in}=C_\text{out}$) and projection shortcuts ($S=2, C_\text{in} \ne C_\text{out}$).
2. Backward pass routes gradient signals through both the residual path $\mathcal{F}$ AND the shortcut path.
3. Finite-difference `gradcheck` verifies analytical `dX` with absolute error $< 10^{-4}$.

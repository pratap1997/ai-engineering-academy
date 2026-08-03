# Module 006: Engineering Challenge — Analytical Conv2D + MaxPool2D Gradcheck

## 1. Challenge Task

Construct a complete, self-contained `ConvPoolBlock` in pure Python & NumPy combining:
1. `Conv2D` layer ($C_\text{in} \rightarrow C_\text{out}$, $K \times K$, $S$, $P$).
2. `MaxPool2D` layer ($2 \times 2$, stride 2).
3. Analytical **backward pass** computing `dX`, `dW`, and `db`.
4. Finite-difference **`gradcheck` verification** matching analytical `dW` and `dX` to within $10^{-4}$ relative/absolute tolerance.

---

## 2. Validation Criteria

1. Forward pass produces correct spatial dimensions $(N, C_\text{out}, H/2, W/2)$.
2. Backward pass routes max pool gradients strictly to the winning activation indices.
3. Finite-difference gradcheck verifies analytical `dW` and `dX` with absolute error $< 10^{-4}$.

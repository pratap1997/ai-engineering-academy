# Module 005: Engineering Challenge — Custom LayerNorm & InvertedDropout Nodes

## 1. Challenge Task

Build a complete, verified PyTorch-compatible `LayerNormDropoutBlock` class in Python that combines:
1. **Layer Normalization** across feature dimensions ($D$).
2. **Inverted Dropout** with configurable probability $p$.
3. Analytical **backward pass** calculation for `dX`, `dgamma`, and `dbeta`.
4. Finite-difference **`gradcheck` verification** matching analytical gradients to within $10^{-5}$ relative tolerance.

---

## 2. Validation Criteria

1. In `train` mode, `forward()` applies LayerNorm followed by Inverted Dropout.
2. In `eval` mode, `forward()` applies LayerNorm and bypasses Dropout.
3. Analytical gradients `dX` pass `gradcheck` tests against finite-difference approximations.

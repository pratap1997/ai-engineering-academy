# Module 018: Engineering Challenge — Causal FlashAttention Implementation

## 1. Challenge Task

Construct a self-contained `CausalFlashAttention` module in pure Python & NumPy that:
1. Implements tiled online softmax with **causal masking** (ensuring $j > i$ tiles are skipped or masked out).
2. Optimizes block processing by skipping Key/Value block tiles where all elements are strictly in the upper triangle ($k_\text{start} > q_\text{end}$).
3. Verifies that output matches standard causal softmax attention to within $10^{-5}$ tolerance.

---

## 2. Validation Criteria

1. Output matches `standard_attention(Q, K, V, causal=True)` to $<10^{-5}$ tolerance.
2. Skipping upper triangular tiles reduces block tile computations by $\approx 50\%$.
3. Zero NaNs or numerical overflow.

# Module 014: Engineering Challenge — RoPE Multi-Head Attention Integration

## 1. Challenge Task

Construct a self-contained `RoPEMultiHeadAttention` module in pure Python & NumPy that:
1. Projects inputs $X \in \mathbb{R}^{N \times T \times d_\text{model}}$ into $Q, K, V$.
2. Applies **RoPE rotation** to $Q$ and $K$ across all heads.
3. Computes Scaled Dot-Product Attention $A = \text{softmax}(Q_\text{rot} K_\text{rot}^T / \sqrt{d_k}) V$.
4. Verifies that shifting a sentence by $+k$ tokens preserves the relative self-attention score matrix up to index displacement.

---

## 2. Validation Criteria

1. Output shape equals input shape $(N, T, d_\text{model})$.
2. Sequence shift invariance: $\text{Attn}(Q_\text{rot}, K_\text{rot})_{m, n} == \text{Attn}(Q'_\text{rot}, K'_\text{rot})_{m+k, n+k}$.
3. Zero NaNs or numerical instabilities.

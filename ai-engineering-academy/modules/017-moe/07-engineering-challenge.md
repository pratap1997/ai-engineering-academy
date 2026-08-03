# Module 017: Engineering Challenge — MoE Transformer Block

## 1. Challenge Task

Construct a self-contained `MoETransformerBlock` in pure Python & NumPy that:
1. Integrates Multi-Head Attention (MHA) with Residual Pre-LN: $X' = X + \text{MHA}(\text{LN}_1(X))$.
2. Integrates `MoELayer` (8 experts, top-2 routing) with Residual Pre-LN: $Y = X' + \text{MoE}(\text{LN}_2(X'))$.
3. Returns output $Y$ and total auxiliary load balancing loss $\mathcal{L}_\text{aux}$.
4. Verifies that output shape matches input shape and zero NaNs occur across batch evaluation.

---

## 2. Validation Criteria

1. Output shape equals $(N, T, d_\text{model})$.
2. Aux loss $> 0.0$ and finite.
3. Zero NaNs or numerical degradation.

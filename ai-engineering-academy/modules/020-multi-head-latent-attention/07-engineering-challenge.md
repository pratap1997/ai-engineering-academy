# Module 020: Engineering Challenge — MLA Autoregressive KV Cache Generator

## 1. Challenge Task

Construct a self-contained `MLAInferenceGenerator` in pure Python & NumPy that:
1. Maintains a compressed KV Cache storing ONLY $\mathbf{c}_{KV}$ (dim $d_c$) and $\mathbf{k}^R$ (dim $d_R$) per token step.
2. Uses Matrix Absorption ($\mathbf{W}_{absorbed} = \mathbf{W}_{UQ} \mathbf{W}_{UK}^T$) to compute content attention scores directly during incremental autoregressive decoding ($T_\text{gen} = 1$).
3. Verifies that incremental generator output matches full-sequence forward pass output up to $10^{-5}$ floating-point tolerance.

---

## 2. Validation Criteria

1. Incremental step-by-step output matches full-sequence causal MLA output.
2. KV Cache size per token equals exactly $d_c + d_R$ elements (zero full-head $n_h \cdot d_h$ storage).
3. Zero NaNs or numerical degradation across generation.

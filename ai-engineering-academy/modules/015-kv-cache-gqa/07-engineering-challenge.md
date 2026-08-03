# Module 015: Engineering Challenge — Incremental Text Generator with GQA & KV Cache

## 1. Challenge Task

Construct a self-contained `IncrementalGQAGenerator` in pure Python & NumPy that:
1. Accepts a prompt sequence $\mathbf{X}_\text{prompt} \in \mathbb{R}^{1 \times T_\text{prompt} \times d_\text{model}}$.
2. Performs **Prefill**: Computes full attention over prompt, populating the `KVCache`.
3. Performs **Decoding**: Generates $N_\text{gen}$ new tokens autoregressively, feeding only 1 token per step into `GroupedQueryAttention` while using `KVCache`.
4. Verifies that output activations generated step-by-step match non-cached full sequence forward pass output.

---

## 2. Validation Criteria

1. KV cache grows monotonically by 1 per step: $\text{len}(K) = T_\text{prompt} + 1, \dots, T_\text{prompt} + N_\text{gen}$.
2. Output activations match batch forward pass to within $10^{-5}$ tolerance.
3. Zero NaNs or dimension mismatches.

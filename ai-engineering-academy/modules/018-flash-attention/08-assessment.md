# Module 018: Assessment & Readiness Check

## 1. Formative Questions

### Q1: Why is standard attention IO-bound rather than compute-bound on modern GPUs?
**Answer**: Standard attention materializes an $O(T^2)$ matrix $\mathbf{P} \in \mathbb{R}^{T \times T}$ in GPU main memory (HBM). At large sequence lengths ($T > 4096$), the GPU spends more time reading and writing the $T \times T$ matrix back and forth over the HBM memory bus ($2\text{ TB/s}$) than actually executing matrix multiplications on fast Tensor Cores ($300\text{--}1000\text{ TFLOPS}$).

### Q2: How does Online Softmax eliminate the need to store the $T \times T$ attention matrix?
**Answer**: Online Softmax maintains running online maximum ($m$) and sum-exp ($d$) statistics as streaming blocks of Key/Value vectors are processed in fast SRAM cache. Rescaling factors $\alpha = e^{m_{\text{old}} - m_{\text{new}}}$ dynamically update partial output accumulators without ever needing to store or re-read previous block attention logits from HBM.

### Q3: Why does FlashAttention recompute attention during the backward pass instead of saving intermediate attention matrices?
**Answer**: In GPU deep learning, memory bandwidth is far more expensive than FLOPS. Recomputing small block tiles of attention on-the-fly in fast SRAM during backward pass is $2\times$--$3\times$ faster than reading pre-computed $O(T^2)$ attention matrices from slow HBM.

---

## 2. Capability Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands HBM vs SRAM memory hierarchy bottlenecks |
| **Competent** | Can explain Online Softmax recurrence relations $(m^\text{new}, d^\text{new})$ |
| **Master** | Can implement `FlashAttentionTiled` forward pass from scratch, implement causal tile skipping, and prove numerical equivalence |

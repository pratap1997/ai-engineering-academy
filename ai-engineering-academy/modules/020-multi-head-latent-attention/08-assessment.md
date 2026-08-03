# Module 020: Assessment & Readiness Check

## 1. Formative Questions

### Q1: Why does Multi-Head Latent Attention (MLA) achieve lower KV Cache footprint than Grouped-Query Attention (GQA)?
**Answer**: GQA reduces KV cache by reducing the number of key/value heads (e.g. 16 heads instead of 128), but still caches uncompressed key/value head vectors. MLA compresses all key and value heads jointly into a single low-rank latent vector $\mathbf{c}_{KV}$ of dimension $d_c$ (e.g., $d_c = 512$ for 128 heads), reducing KV cache size by over $93\%$.

### Q2: What is the Matrix Absorption trick in DeepSeek MLA?
**Answer**: Matrix Absorption pre-computes the matrix product $\mathbf{W}_{\text{absorbed}} = \mathbf{W}_{UQ} \mathbf{W}_{UK}^T$ during model initialization. During inference, attention scores are computed directly between the compressed Query latent vector $\mathbf{c}_Q$ and the compressed Key latent vector $\mathbf{c}_{KV}$, eliminating the need to decompress Keys into full head vectors in VRAM.

### Q3: Why is Rotary Positional Encoding (RoPE) decoupled from content vectors in MLA?
**Answer**: RoPE is position-dependent and dynamic per token step $m$, so it cannot be pre-absorbed into static projection matrices like $\mathbf{W}_{UQ} \mathbf{W}_{UK}^T$. Decoupling RoPE features into a tiny separate vector $\mathbf{k}^R$ allows content features to be fully absorbed while preserving precise rotary spatial positional information.

---

## 2. Capability Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands the difference between MHA, GQA, and MLA |
| **Competent** | Can implement `MLALayer` joint compression into $\mathbf{c}_{KV}$ and decoupled RoPE |
| **Master** | Can implement Matrix Absorption ($\mathbf{W}_{\text{absorbed}} = \mathbf{W}_{UQ} \mathbf{W}_{UK}^T$), build an incremental MLA text generator, and prove numerical equivalence |

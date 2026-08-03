# Module 020: Multi-Head Latent Attention (MLA)

> "Grouped-Query Attention (GQA) reduces KV Cache size by sharing key/value heads across query groups, but sacrifices expressive capacity. Multi-Head Latent Attention (MLA), introduced in DeepSeek-V2 and DeepSeek-V3, compresses Keys and Values into a tiny shared latent vector $c_{KV}$ of dimension $d_c \ll n_h \cdot d_h$, achieving an astonishing **$93\%$ KV cache memory reduction** while matching or exceeding the full attention capacity of Multi-Head Attention (MHA)."

---

## 1. Motivation: The Cache Compression Bottleneck

In standard MHA:
- Key cache size per token: $n_h \cdot d_h$ elements.
- Value cache size per token: $n_h \cdot d_h$ elements.
- Total KV cache per token: $2 \cdot n_h \cdot d_h$ elements.

In **DeepSeek Multi-Head Latent Attention (MLA)**:
1. Keys and Values are jointly compressed into a low-rank latent vector $\mathbf{c}_{KV} \in \mathbb{R}^{d_c}$ ($d_c \ll n_h \cdot d_h$).
2. Decoupled RoPE features $\mathbf{k}_{R} \in \mathbb{R}^{d_R}$ are cached separately to preserve rotary spatial positional embeddings.
3. Total KV cache cached per token during inference is only **$d_c + d_R$ elements**!

---

## 2. Matrix Absorption Trick (Inference Optimization)

During training:
$$\mathbf{K}^C = \mathbf{c}_{KV} \mathbf{W}_{UK}, \quad \mathbf{V}^C = \mathbf{c}_{KV} \mathbf{W}_{UV}$$

During inference, multiplying $\mathbf{c}_{KV}$ by $\mathbf{W}_{UK}$ to restore full Key heads would re-expand memory overhead. DeepSeek solves this by **absorbing** $\mathbf{W}_{UK}$ into Query projection matrix $\mathbf{W}_{UQ}$:

$$\mathbf{Q}^C \mathbf{K}^{CT} = (\mathbf{c}_Q \mathbf{W}_{UQ}) (\mathbf{c}_{KV} \mathbf{W}_{UK})^T = \mathbf{c}_Q (\mathbf{W}_{UQ} \mathbf{W}_{UK}^T) \mathbf{c}_{KV}^T$$

This allows the GPU to compute attention scores directly between compressed Query representations and the tiny compressed latent vector $\mathbf{c}_{KV}$ without ever decompressing Keys or Values in VRAM!

---

## 3. MHA vs GQA vs MLA Comparison

| Metric | Multi-Head Attention (MHA) | Grouped-Query Attention (GQA) | Multi-Head Latent Attention (MLA) |
|---|---|---|---|
| **KV Cache per Token** | $2 \cdot n_h \cdot d_h$ | $2 \cdot n_{kv} \cdot d_h$ | $\mathbf{d_c + d_R}$ |
| **DeepSeek-V3 Specs** | $128 \cdot 128 = 16,384$ | $16 \cdot 128 = 2,048$ | $\mathbf{512 + 64 = 576}$ ($93.3\%$ reduction!) |
| **Model Expressivity** | $100\%$ (Full rank) | Reduced rank | **Full MHA Rank Preserved** |

---

## 4. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → The zipped folder archive & matrix absorption trick
03-mathematics.md       → Low-rank joint compression & decoupled RoPE equations
04-implementation.py    → MLALayer, MatrixAbsorption, KVCacheCompressed
05-experiments.py       → KV cache size benchmark (93% compression) & Matrix absorption speedup
06-real-applications.md → DeepSeek-V2, DeepSeek-V3, DeepSeek-R1 architecture integration
07-engineering-challenge.md → MLA Autoregressive Text Generation with Absorbed Matrices
08-assessment.md        → Readiness check
09-references.md        → DeepSeek-AI (2024)
```

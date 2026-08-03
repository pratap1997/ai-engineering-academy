# Module 018: FlashAttention & Tiled Online-Softmax Algorithm

> "Standard attention materializes an $O(T^2)$ attention matrix in GPU High Bandwidth Memory (HBM), causing massive memory reads and writes. FlashAttention restructures attention into block-wise SRAM tiles using Online Softmax, reducing HBM memory access by $5\times$--$10\times$ and speeding up training by $2\times$--$4\times$ with ZERO loss in mathematical precision."

---

## 1. Motivation: The IO Memory Wall

In standard Multi-Head Attention:

$$\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}, \quad \mathbf{P} = \text{softmax}(\mathbf{S}) \in \mathbb{R}^{T \times T}, \quad \mathbf{O} = \mathbf{P} \mathbf{V} \in \mathbb{R}^{T \times d_d}$$

For sequence length $T = 16,384$:
- Storing $P \in \mathbb{R}^{16384 \times 16384}$ requires **512 MB per head per layer**.
- For 32 layers and 32 heads, storing attention matrices in HBM requires **524 GB VRAM**!

**FlashAttention** (Tri Dao et al., 2022) introduces an **IO-aware** algorithm that:
1. Splits $Q, K, V$ into small block tiles that fit inside GPU fast SRAM cache ($64 \text{ KB}$--$256 \text{ KB}$).
2. Uses **Online Softmax** to accumulate max values and sum-exp statistics incrementally without materializing the full $T \times T$ matrix in HBM.
3. Recomputes attention tiles on-the-fly during the backward pass instead of storing $P$ in memory.

---

## 2. Standard Attention vs FlashAttention

| Aspect | Standard Attention | FlashAttention |
|---|---|---|
| **HBM Memory Usage** | $O(T^2)$ | $O(T)$ |
| **HBM Access (Bytes Read/Written)** | $O(T^2 + T \cdot d)$ | $O(T^2 \cdot d^2 / M_{\text{SRAM}})$ |
| **Wall-Clock Speedup** | $1\times$ (Baseline) | $2\times$--$4\times$ Faster |
| **Precision** | Exact | Exact (Identical output to 1e-6) |

---

## 3. Learning Outcomes

1. **Implement `OnlineSoftmax`**: Accumulate softmax streaming statistics $(m_{\text{new}}, d_{\text{new}})$ incrementally.
2. **Implement `FlashAttentionTiled`**: Block-wise matrix multiplication over $B_r \times B_c$ SRAM tiles.
3. **Analyze GPU Memory Hierarchy**: Explain HBM (High Bandwidth Memory) vs SRAM (Static RAM) latency and bandwidth trade-offs.
4. **Verify Exact Equivalence**: Prove that FlashAttention output matches standard softmax attention up to floating-point tolerance.

---

## 4. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → The assembly line conveyor belt & SRAM tiles
03-mathematics.md       → Online Softmax recurrence relations & Tiled FlashAttention loop
04-implementation.py    → OnlineSoftmax, FlashAttentionTiled
05-experiments.py       → Peak memory benchmark (O(T) vs O(T^2)) & Block size sweep
06-real-applications.md → FlashAttention-2, FlashAttention-3 (Hopper TMA), PyTorch scaled_dot_product_attention
07-engineering-challenge.md → FlashAttention with Causal Masking
08-assessment.md        → Readiness check
09-references.md        → Dao et al. (2022, 2023)
```

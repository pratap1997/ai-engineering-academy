# Module 020: Real Applications — DeepSeek-V2/V3 & DeepSeek-R1 Architecture

## 1. DeepSeek-V3 Production Specs

DeepSeek-V3 employs MLA across all 61 Transformer layers:

```
d_model:     7168
num_heads:   128
d_h (head):  128
d_c (latent): 512   (Joint Key-Value compression dimension)
d_R (RoPE):   64    (Decoupled RoPE dimension)
KV Cache per token: 576 FP16 numbers (1,152 Bytes) vs 32,768 Bytes for MHA!
```

---

## 2. Serving 128k Context with 93% Less VRAM

At $128,000$ token context length:
- Standard MHA KV Cache per user: **4.19 GB VRAM**
- DeepSeek MLA KV Cache per user: **0.147 GB VRAM** (147 MB!)

This single architectural breakthrough enables DeepSeek to serve **$28\times$ more concurrent user requests per GPU node** than competing models!

---

## 3. MLA Integration in DeepSeek-R1

DeepSeek-R1 combines:
1. **MLA (Multi-Head Latent Attention)** for ultra-low VRAM KV Cache footprint.
2. **DeepSeekMoE (256 fine-grained experts)** for sparse compute efficiency.
3. **GRPO (Group Relative Policy Optimization)** for reasoning reinforcement learning.

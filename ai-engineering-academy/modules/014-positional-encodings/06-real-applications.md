# Module 014: Real Applications — Production Positional Encodings

## 1. LLaMA 3 & RoPE Scaling (YaRN)

Modern frontier models (LLaMA 3, Mistral 7B) use **RoPE** as their primary position encoding mechanism.

```python
import torch

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (batch_size, seq_len, num_heads, head_dim)
    # cos, sin: (1, seq_len, 1, head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

### Context Window Scaling with YaRN / Linear Scaling
To extend LLaMA 2 from 4K context to 32K/128K context:
- **Linear Scaling**: Divide frequencies $\theta_i$ by scale factor $s = 8$ or $32$.
- **YaRN (Yet Another RoPE Extension)**: Dynamically scales high and low frequency dimensions differently, preserving short-range resolution while enabling long-range context without re-training from scratch.

---

## 2. ALiBi in MPT & BLOOM

MPT-7B and BLOOM use **ALiBi**, which allows training on 2,048 tokens and evaluating seamlessly on 8,192+ tokens at inference time with zero additional fine-tuning.

```python
# PyTorch ALiBi attention score modification
attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
attn_weights = attn_weights + alibi_bias  # (batch, num_heads, seq_len, seq_len)
attn_weights = torch.softmax(attn_weights, dim=-1)
```

---

## 3. Position Encoding Method Comparison

| Model | Technique | Extrapolation Ability | Extra Parameters | Compute Overhead |
|---|---|---|---|---|
| **Transformer (2017)** | Sinusoidal Absolute | Low | 0 | Negligible |
| **BERT / GPT-2** | Learned Absolute | Zero | $|V| \cdot d$ | Negligible |
| **T5** | Relative Bias | High | $N_\text{buckets} \cdot H$ | Small |
| **LLaMA 1-3 / Mistral** | RoPE | Very High (with YaRN) | 0 | Minimal (cos/sin mul) |
| **MPT / BLOOM** | ALiBi | Maximum | 0 | Zero |

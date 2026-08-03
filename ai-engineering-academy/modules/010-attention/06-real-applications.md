# Module 010: Real Applications — BERT, GPT & Production Attention

## 1. PyTorch Multi-Head Attention

```python
import torch
import torch.nn as nn

# nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True, dropout=0.1)

X = torch.randn(32, 50, 512)  # (N, T, d_model)

# Self-attention: Q = K = V = X
output, attn_weights = mha(X, X, X)
# output: (32, 50, 512)
# attn_weights: (32, 50, 50)

# Causal mask for autoregressive generation (GPT-style):
T = 50
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
output_causal, _ = mha(X, X, X, attn_mask=causal_mask)
```

---

## 2. Three Attention Variants

| Variant | Q source | K,V source | Used In |
|---|---|---|---|
| **Self-Attention** | Input sequence | Input sequence | BERT Encoder, GPT Decoder |
| **Cross-Attention** | Decoder state | Encoder output | Translation Seq2Seq |
| **Causal Self-Attention** | Input (masked future) | Input (past only) | GPT, language generation |

---

## 3. BERT vs GPT Attention

| | **BERT** | **GPT** |
|---|---|---|
| **Attention Type** | Bidirectional Self-Attention | Causal (Masked) Self-Attention |
| **Can See** | Past + Future tokens | Past tokens only |
| **Use Case** | Understanding (classification, NER) | Generation (text completion) |
| **Training** | Masked Language Model (MLM) | Next-token prediction |

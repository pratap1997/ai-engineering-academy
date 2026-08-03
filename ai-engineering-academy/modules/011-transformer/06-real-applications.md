# Module 011: Real Applications

## 1. BERT-base Architecture

```
BERT-base = TransformerEncoder(
    d_model    = 768,
    num_heads  = 12,
    num_layers = 12,
    d_ff       = 3072  # 4 × 768
)
Total params ≈ 110 million
```

## 2. GPT-2 Small Architecture

```
GPT-2 Small = TransformerDecoder(
    d_model    = 768,
    num_heads  = 12,
    num_layers = 12,
    d_ff       = 3072
)
Key difference: Causal (masked) self-attention only — no cross-attention.
Total params ≈ 117 million
```

## 3. PyTorch One-Liner

```python
import torch.nn as nn

# One Encoder block
block = nn.TransformerEncoderLayer(
    d_model=512, nhead=8, dim_feedforward=2048,
    dropout=0.1, activation='gelu', norm_first=True  # Pre-LN
)
encoder = nn.TransformerEncoder(encoder_layer=block, num_layers=6)

# x: (batch, seq_len, d_model) with batch_first=True
output = encoder(x)
```

## 4. BERT vs GPT Side-by-Side

| | BERT | GPT |
|---|---|---|
| **Block type** | Encoder | Decoder |
| **Attention mask** | Bidirectional | Causal (upper triangle) |
| **Pre-training** | Masked LM + NSP | Next-token prediction |
| **Best for** | Classification, NER, QA | Text generation |

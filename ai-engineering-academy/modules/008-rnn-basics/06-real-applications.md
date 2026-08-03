# Module 008: Real Applications — Sequence Taxonomy & Production Recurrent Layers

## 1. Sequence Modeling Taxonomy

RNNs handle different sequence input/output configurations:

| Taxonomy | Input Shape | Output Shape | Real-World Application |
|---|---|---|---|
| **One-to-One** | Single Vector | Single Vector | Standard Image Classification |
| **One-to-Many** | Single Vector | Sequence $T$ | Image Captioning (Image $\rightarrow$ "A cat sitting on a mat") |
| **Many-to-One** | Sequence $T$ | Single Vector | Sentiment Analysis ("This movie is great" $\rightarrow$ Positive) |
| **Many-to-Many (Synced)** | Sequence $T$ | Sequence $T$ | Video Frame Labeling, Token Tagging |
| **Many-to-Many (Seq2Seq)** | Sequence $T_1$ | Sequence $T_2$ | Machine Translation (English $\rightarrow$ French) |

---

## 2. PyTorch Syntax & Truncated BPTT

```python
import torch
import torch.nn as nn

# nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

# Input X: (batch_size, seq_len, input_size)
X = torch.randn(32, 50, 10)
output, h_n = rnn(X)

# output shape: (32, 50, 20) -> All hidden states across time
# h_n shape:    (1, 32, 20)  -> Final hidden state at T
```

---

## 3. Truncated BPTT for Long Streams

When processing sequences with thousands of tokens (e.g. books or continuous audio), unrolling the entire sequence for backpropagation is computationally impossible.

> 💡 **Truncated BPTT**: Break the long stream into smaller chunks of length $T_\text{chunk} = 50$. Pass hidden state $\mathbf{h}_{T}$ forward to the next chunk as an un-differentiable initial state `h.detach()`.

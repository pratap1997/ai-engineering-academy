# Module 009: Real Applications — Production Gated Networks

## 1. PyTorch LSTM & GRU Syntax

```python
import torch
import torch.nn as nn

# PyTorch LSTM Layer: (input_size, hidden_size, num_layers, batch_first=True)
lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)

# Input X: (N, T, D) -> (32, 50, 64)
X = torch.randn(32, 50, 64)

# Forward returns: output, (h_n, c_n)
output, (h_n, c_n) = lstm(X)

print("Output Shape:", output.shape) # (32, 50, 128)
print("h_n Shape:", h_n.shape)       # (2, 32, 128) -> (num_layers, batch, hidden)
print("c_n Shape:", c_n.shape)       # (2, 32, 128) -> Final Cell State
```

---

## 2. Bidirectional LSTMs (BiLSTM)

In tasks where future context is available (e.g. named entity recognition, speech processing):
- **Forward LSTM**: Reads sequence left-to-right ($t = 1 \rightarrow T$).
- **Backward LSTM**: Reads sequence right-to-left ($t = T \rightarrow 1$).
- **Concatenated Representation**: $\mathbf{h}_t^\text{BiLSTM} = [\mathbf{h}_t^\rightarrow ; \mathbf{h}_t^\leftarrow]$.

---

## 3. When to Choose LSTM vs GRU vs Transformer

1. **Use GRU**: Small datasets, fast iteration, constrained edge devices ($25\%$ fewer parameters than LSTM).
2. **Use LSTM**: Complex long sequences with strong forget/remember dynamics.
3. **Use Transformer**: Large parallel datasets, long contexts ($T > 512$), where non-sequential GPU parallelization is required.

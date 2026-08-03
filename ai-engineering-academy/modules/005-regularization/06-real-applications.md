# Module 005: Real Applications — Production Regularization & Normalization

## 1. Modern Framework Equivalents

| Module 005 Component | PyTorch Class / Argument | Key Parameters |
|---|---|---|
| **L2 Regularization** | `weight_decay` in PyTorch optimizers | `weight_decay=1e-4` |
| **Inverted Dropout** | `torch.nn.Dropout(p=0.5)` | `p=0.5` |
| **Batch Normalization** | `torch.nn.BatchNorm1d(num_features)` | `eps=1e-5`, `momentum=0.1` |
| **Layer Normalization** | `torch.nn.LayerNorm(normalized_shape)` | `eps=1e-5`, `elementwise_affine=True` |
| **RMSNorm** | Modern Transformer Alternative (LLaMA) | Ignores mean shift $\mu$, normalizes by RMS only for $7\%$ speedup |

---

## 2. Industry Best Practices

### Rule 1: Never apply L2 Weight Decay to Normalization Parameters ($\gamma, \beta$) or Biases
In PyTorch training loops, parameters are grouped:
- **Weights ($\mathbf{W}$)**: Subject to `weight_decay`.
- **Biases ($\mathbf{b}$) and Norm Parameters ($\boldsymbol{\gamma}, \boldsymbol{\beta}$)**: `weight_decay = 0.0`.

### Rule 2: Where to place Normalization in Transformer Networks
- **Post-LN** (Original Transformer 2017): Norm placed *after* residual addition $\mathbf{x} + \text{SubLayer}(\text{Norm}(\mathbf{x}))$. Requires warmups.
- **Pre-LN** (Modern standard, GPT-3/LLaMA): Norm placed *before* sub-layer $\mathbf{x} + \text{SubLayer}(\text{Norm}(\mathbf{x}))$. Stable without warmups.

### Rule 3: Always set `.train()` and `.eval()` Modes
Functions like Dropout and BatchNorm behave differently in training vs evaluation modes. Forgetting `model.eval()` before running test inference is a common cause of silent production bugs!

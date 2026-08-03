# Module 004: Real Applications — Production Training Infrastructure

## 1. PyTorch & Framework Equivalents

The pure Python/NumPy classes you implemented in `04-implementation.py` directly mirror production AI frameworks:

| Module 004 Class | PyTorch Equivalent | Key Hyperparameters |
|---|---|---|
| `SGD` | `torch.optim.SGD(params, lr)` | `lr` |
| `Momentum` | `torch.optim.SGD(params, lr, momentum=0.9)` | `lr`, `momentum`, `dampening`, `nesterov` |
| `RMSprop` | `torch.optim.RMSprop(params, lr, alpha=0.99)` | `lr`, `alpha`, `eps`, `weight_decay` |
| `Adam` | `torch.optim.Adam(params, lr, betas=(0.9, 0.999))` | `lr`, `betas`, `eps` |
| `AdamW` | `torch.optim.AdamW(params, lr, weight_decay=0.01)` | `lr`, `betas`, `eps`, `weight_decay` |
| `CosineAnnealingLR` | `torch.optim.lr_scheduler.CosineAnnealingLR` | `T_max`, `eta_min` |

---

## 2. Industry Standard Training Recipes

### Recipe 1: Computer Vision (ResNet, ViT)
- **Optimizer**: `SGD` with `momentum=0.9` OR `AdamW(lr=1e-3, weight_decay=0.05)`
- **Schedule**: Cosine Annealing with 5-epoch Linear Warmup.
- **Batch Size**: 128 to 1024.

### Recipe 2: Large Language Models & Transformers (LLaMA, GPT-4)
- **Optimizer**: `AdamW(beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1)`
- **Gradient Clipping**: Max norm = 1.0 (prevents Transformer attention spikes).
- **Schedule**: Linear Warmup + Cosine Decay to $10\%$ initial learning rate.

### Recipe 3: Tabular Data (MLPs)
- **Optimizer**: `Adam(lr=1e-3)`
- **Regularization**: Dropout (0.2) + Early Stopping (`patience=10`).

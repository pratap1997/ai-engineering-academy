# Module 007: Real Applications — Industrial ResNet Specifications

## 1. Official ResNet Family Specifications

| Model | Block Type | Layer Config | FLOPS | Top-1 Accuracy |
|---|---|---|---|---|
| **ResNet-18** | Basic ($3 \times 3, 3 \times 3$) | $[2, 2, 2, 2]$ | $1.8 \text{ GFLOPS}$ | $69.8\%$ |
| **ResNet-34** | Basic ($3 \times 3, 3 \times 3$) | $[3, 4, 6, 3]$ | $3.6 \text{ GFLOPS}$ | $73.3\%$ |
| **ResNet-50** | Bottleneck ($1 \times 1, 3 \times 3, 1 \times 1$) | $[3, 4, 6, 3]$ | $4.1 \text{ GFLOPS}$ | $76.1\%$ |
| **ResNet-101** | Bottleneck ($1 \times 1, 3 \times 3, 1 \times 1$) | $[3, 4, 23, 3]$ | $7.8 \text{ GFLOPS}$ | $77.4\%$ |
| **ResNet-152** | Bottleneck ($1 \times 1, 3 \times 3, 1 \times 1$) | $[3, 8, 36, 3]$ | $11.6 \text{ GFLOPS}$ | $78.3\%$ |

---

## 2. Modern Adaptations: ConvNeXt & Transformer ResNets

In 2022, Liu et al. introduced **ConvNeXt**, modernizing ResNets using lessons learned from Vision Transformers (ViT):
1. **$7 \times 7$ Depthwise Separable Convolutions**: Increased receptive field per layer.
2. **Inverted Bottleneck**: Expanded channel dimension in middle sub-layers ($C \rightarrow 4C \rightarrow C$).
3. **GELU Activations & LayerNorm**: Replaced ReLU and BatchNorm with LayerNorm and GELU.

---

## 3. PyTorch `torchvision` Usage

```python
import torchvision.models as models

# Load pretrained ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Replace final classification head for custom 10-class task
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
```

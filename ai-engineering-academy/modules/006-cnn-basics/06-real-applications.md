# Module 006: Real Applications — Production CNN Architectures

## 1. Landmark CNN Architectures

| Architecture | Year | Key Breakthrough | Parameters |
|---|---|---|---|
| **LeNet-5** | 1998 | First commercial CNN for digit recognition (MNIST) | ~60,000 |
| **AlexNet** | 2012 | ImageNet breakthrough using GPUs, ReLU, and Dropout | ~60 Million |
| **VGG-16** | 2014 | Standardized $3 \times 3$ convolutions stacked deep | ~138 Million |
| **ResNet-50** | 2015 | Residual connections allowing 50–152 layer depth | ~25 Million |

---

## 2. PyTorch Equivalents & Syntax

```python
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
```

---

## 3. Key Design Rules for Modern CNNs

1. **Prefer $3 \times 3$ Kernels**: Two stacked $3 \times 3$ conv layers have a receptive field of $5 \times 5$, but use $18 C^2$ parameters instead of $25 C^2$ parameters ($28\%$ savings), plus 2 non-linearities instead of 1.
2. **Double Channels When Downsampling**: When spatial resolution is halved via pooling or stride-2 convolution ($H \rightarrow H/2, W \rightarrow W/2$), double the channel count ($C \rightarrow 2C$) to preserve feature volume representation capacity.

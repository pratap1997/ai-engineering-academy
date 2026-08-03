# Module 006: Convolutional Neural Networks (Kernels, Convolutions & Pooling)

> "A dense linear layer treats every input pixel independently, ignoring whether two pixels are side-by-side or on opposite ends of an image. A Convolutional layer bakes spatial geometry directly into network architecture through parameter sharing and local receptive fields."

---

## 1. Motivation: Spatial Inductive Bias & Parameter Efficiency

Consider classifying a $256 \times 256$ RGB image ($196,608$ input dimensions) using a single Dense layer with $1,000$ hidden units:
- **Dense Layer Parameter Count**: $196,608 \times 1,000 \approx 196.6 \text{ Million Weights}$!
- **Flaws**: Extreme overfitting, no translation invariance (a cat in the top-left corner requires learning completely different weights than a cat in the bottom-right corner).

**Convolutional Neural Networks (CNNs)** solve this by introducing two spatial priors:
1. **Local Connectivity**: Each output unit only looks at a small $K \times K$ patch of the input (receptive field).
2. **Parameter Sharing**: The exact same filter weights ($K \times K$) slide across the entire image, searching for a specific feature (edges, textures, corners) everywhere.

---

## 2. Learning Outcomes

By completing this module, you will be able to:

1. **Implement 2D Convolution**: Build `Conv2D` with multi-channel support ($C_\text{in} \rightarrow C_\text{out}$), padding ($P$), and stride ($S$) from scratch in pure Python & NumPy.
2. **Implement Pooling Layers**: Build `MaxPool2D` and `AvgPool2D` with forward indexing masks for backward gradient routing.
3. **Calculate Spatial Math**: Derive exact formulas for output dimensions and receptive field expansion across multi-layer CNNs.
4. **Perform Analytical CNN Backpropagation**: Derive and verify the backward pass for 2D convolutions using spatial cross-correlation and transposed convolution.

---

## 3. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → Sliding window, Sobel edge filters, & pooling abstraction
03-mathematics.md       → Output size formula, receptive field math, & Conv2D backward calculus
04-implementation.py    → Pure Python & NumPy implementations of Conv2D, MaxPool2D, AvgPool2D, Flatten
05-experiments.py       → Sobel filter edge detection, receptive field growth, & parameter efficiency benchmark
06-real-applications.md → LeNet-5, AlexNet, VGG, ResNet stem, & PyTorch torch.nn.Conv2d
07-engineering-challenge.md → Custom Conv2D + MaxPool2D block with analytical backward pass & gradcheck
08-assessment.md        → Readiness check & self-assessment rubrics
09-references.md        → LeCun et al. (1998) & Krizhevsky et al. (2012) citations
```

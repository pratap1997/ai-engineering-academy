# Module 005: Regularization, Normalization & Overfitting

> "A neural network with high capacity can easily memorize training noise. Regularization constrains the network's capacity to learn simple patterns, while Normalization stabilizes internal signals so deep architectures can actually learn."

---

## 1. Motivation: Generalization & Internal Covariate Shift

In **Modules 001–004**, we trained neural networks to minimize training loss. However, the ultimate goal of machine learning is not training performance — it is **generalization to unseen test data**.

When a model has more capacity than required for a task:
1. **Overfitting**: It memorizes noise and outliers, resulting in near-zero training error but high test error.
2. **Internal Covariate Shift**: As early layers update their weights during training, the input distribution to deeper layers constantly shifts, slowing down convergence and causing gradient instability.

**Module 005** provides the complete engineering toolkit to solve both problems:
- **Regularization**: L1 (Lasso / Sparsity), L2 (Ridge / Weight Decay), and Inverted Dropout.
- **Normalization**: Batch Normalization (BatchNorm1d) and Layer Normalization (LayerNorm).

---

## 2. Learning Outcomes

By completing this module, you will be able to:

1. **Implement Regularization**: Build L1, L2, and Inverted Dropout layers from scratch in pure Python & NumPy.
2. **Implement Normalization**: Build `BatchNorm1d` (with running mean/variance tracking for train vs eval modes) and `LayerNorm` (with learnable scale $\gamma$ and shift $\beta$).
3. **Compare L1 vs L2 Geometrically**: Prove why L1 regularization forces exact zero weights (feature selection / sparsity) while L2 scales weights down smoothly.
4. **Build a Regularized Pipeline**: Construct a training pipeline that prevents overfitting on high-variance data and achieves low test error.

---

## 3. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → Geometry of L1/L2 contours, Dropout ensemble, & Normalization
03-mathematics.md       → Mathematical derivations of L1/L2 gradients, Dropout, BatchNorm & LayerNorm
04-implementation.py    → Pure Python & NumPy implementations of Dropout, BatchNorm1d, LayerNorm, and RegularizedMLP
05-experiments.py       → L1 Sparsity experiment, BatchNorm convergence speedup, Overfitting vs Generalization curves
06-real-applications.md → Modern normalization in Transformers (LayerNorm / RMSNorm in LLaMA) and CNNs (BatchNorm)
07-engineering-challenge.md → Custom LayerNorm & InvertedDropout nodes with gradcheck
08-assessment.md        → Readiness check & self-assessment rubrics
09-references.md        → Srivastava (2014), Ioffe & Szegedy (2015), Ba et al. (2016) citations
```

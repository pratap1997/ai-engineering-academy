# Module 022: Knowledge Distillation & Student Model Extraction

> "Deploying a massive 70B parameter Teacher model for edge inference is economically impractical. Knowledge Distillation (Geoffrey Hinton et al., 2015) transfers dark knowledge—the rich probability distributions over incorrect classes—from a high-capacity Teacher to a compact Student model, enabling a $60\%$ smaller model to retain $97\%$ of Teacher capability."

---

## 1. Motivation: Dark Knowledge & Soft Targets

When a Teacher model evaluates an image of a BMW car, hard ground-truth labels say:
- `[Car: 1, Truck: 0, Bicycle: 0, House: 0]`

However, the Teacher model's un-normalized logit distribution reveals **dark knowledge**:
- `[Car: 0.85, Truck: 0.12, Bicycle: 0.02, House: 0.0001]`

The Teacher implicitly encodes that a Car looks somewhat like a Truck, but nothing like a House!

**Temperature Softening**:
By scaling logits by temperature $T > 1$:

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

As $T$ increases, dark knowledge probability ratios become prominent, providing rich gradient signals for the Student model to learn dark structural relationships!

---

## 2. The Distillation Loss Formulation

The total Distillation Loss combines:
1. **KL Divergence Loss** between softened Teacher distribution $\mathbf{q}^T$ and softened Student distribution $\mathbf{p}^T$.
2. **Cross-Entropy Loss** between Student hard predictions and ground-truth one-hot labels $\mathbf{y}$.

$$\mathcal{L}_{\text{Distill}} = \alpha \cdot T^2 \cdot D_{\text{KL}}(\mathbf{q}^T \parallel \mathbf{p}^T) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(\mathbf{y}, \mathbf{p})$$

Where $T^2$ balances gradient magnitude when temperature scaling is applied.

---

## 3. Teacher vs Student Trade-Offs (DistilBERT Case Study)

| Model | Layers | Parameters | Inference Latency | Benchmark Performance Retained |
|---|---|---|---|---|
| **BERT-Base (Teacher)** | 12 | 110 Million | 100 ms (Baseline) | $100\%$ |
| **DistilBERT (Student)** | 6 | 66 Million | 40 ms (**$2.5\times$ Faster**) | **$97\%$ Retained** |

---

## 4. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → The master professor & apprentice notebook
03-mathematics.md       → Temperature-scaled Softmax & KL-divergence gradient derivation
04-implementation.py    → SoftmaxWithTemperature, KLDivergenceLoss, KnowledgeDistiller
05-experiments.py       → Temperature T sweep (T=1, 2, 4, 8) & Student compression convergence
06-real-applications.md → DistilBERT, DeepSeek-R1 Distilled Qwen Models (1.5B - 70B)
07-engineering-challenge.md → Distilling 12-layer Teacher to 4-layer Student
08-assessment.md        → Readiness check
09-references.md        → Hinton et al. (2015), Sanh et al. (2019)
```

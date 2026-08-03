# Module 019: Parameter-Efficient Fine-Tuning (LoRA & QLoRA)

> "Fine-tuning a 70B parameter model by updating all 70B weights (full fine-tuning) requires optimizer state VRAM of over $1.4\text{ TB}$. LoRA freezes base model weights and trains two tiny low-rank matrices $A$ and $B$ containing $<0.1\%$ of total parameters, reducing VRAM by $10\times$ while matching full fine-tuning performance."

---

## 1. Motivation: The Cost of Full Fine-Tuning

During AdamW training, for every parameter $w$:
- Weight $w$: 2 bytes (FP16)
- Gradient $g$: 2 bytes (FP16)
- First momentum $m$: 4 bytes (FP32)
- Second momentum $v$: 4 bytes (FP32)
- Total per parameter: **12--16 bytes**

For a 70B model:
$$\text{VRAM}_{\text{Full Fine-Tuning}} = 70 \times 10^9 \times 16 \text{ bytes} \approx \mathbf{1.12 \text{ TB VRAM}}$$

**Low-Rank Adaptation (LoRA)**:
Freeze base weight $\mathbf{W}_0 \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$.
Inject trainable rank-$r$ decomposition matrices $\mathbf{A} \in \mathbb{R}^{r \times d_\text{in}}$ and $\mathbf{B} \in \mathbb{R}^{d_\text{out} \times r}$ ($r \ll d$):

$$\Delta \mathbf{W} = \frac{\alpha}{r} \mathbf{B} \mathbf{A}$$

$$\mathbf{h} = \mathbf{x} \mathbf{W}_0^T + \frac{\alpha}{r} \mathbf{x} \mathbf{A}^T \mathbf{B}^T$$

---

## 2. LoRA vs QLoRA

| Method | Base Weights ($\mathbf{W}_0$) | Trainable Adapter ($\mathbf{A}, \mathbf{B}$) | Fine-tuning 70B VRAM |
|---|---|---|---|
| **Full Fine-Tuning** | FP16 (Trainable) | N/A | $1,120 \text{ GB}$ |
| **LoRA** | FP16 (Frozen) | FP16/FP32 (Trainable) | $140 \text{ GB}$ |
| **QLoRA** | INT4 / NF4 (Frozen) | FP16/FP32 (Trainable) | $\mathbf{48 \text{ GB}}$ (Single GPU!) |

---

## 3. Zero-Latency Inference Merging

During inference, LoRA introduces zero latency penalty.
Before deployment, merge adapter weights directly into the base weights:

$$\mathbf{W}_{\text{merged}} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}$$

The resulting model is a standard single-weight linear layer with **zero additional compute or memory overhead** during serving!

---

## 4. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → The translucent plastic overlay sheet analogy
03-mathematics.md       → Low-rank matrix decomposition & scaling factor alpha/r
04-implementation.py    → LoRALinear, QLoRALinear, merge/unmerge functionality
05-experiments.py       → Parameter count reduction (>99.9%) & weight merging latency benchmark
06-real-applications.md → HuggingFace PEFT, QLoRA NF4 config, adapter merging
07-engineering-challenge.md → LoRA Multi-Head Attention layer with adapter switching
08-assessment.md        → Readiness check
09-references.md        → Hu et al. (2021), Dettmers et al. (2023)
```

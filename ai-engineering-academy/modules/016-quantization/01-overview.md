# Module 016: Model Quantization (INT8, INT4 Weight-Only & Scaling)

> "Modern 70B parameter models require 140 GB VRAM in FP16, exceeding single GPU capacities. Quantization compresses 16-bit floating point weights into 8-bit or 4-bit integers with $<0.5\%$ loss in accuracy, reducing memory footprint by $4\times$--$8\times$ and enabling enterprise LLM serving on consumer hardware."

---

## 1. Motivation: The VRAM Wall

A 70B parameter LLM stored in 16-bit floating point (FP16/BF16) requires:

$$70 \times 10^9 \times 2 \text{ bytes} = 140 \text{ GB VRAM}$$

This requires two $80\text{ GB}$ NVIDIA H100 GPUs just to load the model parameters into memory!

**Quantization** maps continuous floating-point values $w \in [\min, \max]$ to discrete integer values $q \in [q_{\min}, q_{\max}]$:
- **INT8 Quantization**: $4\times$ memory reduction ($140\text{ GB} \to 35\text{ GB}$). Fits on a single A100 GPU!
- **INT4 Weight-Only Quantization**: $8\times$ memory reduction ($140\text{ GB} \to 17.5\text{ GB}$). Fits on a single consumer RTX 3090 / 4090 GPU!

---

## 2. Quantization Paradigms

1. **Symmetric Quantization**: Zero-point $z = 0$. Range is centered around zero. Simpler math, ideal for zero-centered weights.
2. **Asymmetric Quantization**: Zero-point $z \neq 0$. Maps $[\min, \max]$ directly to $[q_{\min}, q_{\max}]$. Preserves precision when activation distributions are non-negative (e.g., ReLU/GELU).
3. **Block-wise / Group-wise Quantization**: Computes separate scale factors $s$ for groups of 64 or 128 weights, preserving accuracy for INT4 quantization.

---

## 3. Learning Outcomes

1. **Implement `SymmetricQuantizer`**: INT8 & INT4 scale calculation and quantization/dequantization.
2. **Implement `AsymmetricQuantizer`**: Scale $s$ and zero-point $z$ computation.
3. **Implement `GroupQuantizer`**: Block-wise INT4 weight quantization.
4. **Implement `QuantizedLinear`**: Quantized weight linear layer with dequantization on-the-fly.

---

## 4. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → The grid ruler analogy & scaling factor maps
03-mathematics.md       → Scale, zero-point, quantization, dequantization equations
04-implementation.py    → SymmetricQuantizer, AsymmetricQuantizer, GroupQuantizer, QuantizedLinear
05-experiments.py       → Memory compression ratio benchmark & MSE error vs bit width
06-real-applications.md → GPTQ, AWQ, bitsandbytes (LLM.int8), SmoothQuant
07-engineering-challenge.md → Quantized Multi-Head Attention layer
08-assessment.md        → Readiness check
09-references.md        → Dettmers et al. (2022), Frantar et al. (2022)
```

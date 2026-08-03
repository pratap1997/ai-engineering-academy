# Module 016: Assessment & Readiness Check

## 1. Formative Questions

### Q1: Why does asymmetric quantization require a zero-point $z$?
**Answer**: Asymmetric quantization maps the arbitrary floating-point range $[\min(X), \max(X)]$ directly to the full unsigned range $[0, 255]$. Because real-world zero ($0.0$) may not map to integer index 0, $z$ acts as an offset to ensure that real $0.0$ corresponds exactly to integer $z$.

### Q2: Why is per-channel / group-wise quantization critical for INT4?
**Answer**: In 4-bit quantization, there are only 16 discrete levels ($[-7, 7]$ or $[0, 15]$). If a single scale factor is used for a whole 2D weight matrix, a single large outlier weight compresses all other weights into 1 or 2 discrete levels. Grouping weights into blocks of 64 or 128 isolates outliers to single groups, preserving precision across the remaining matrix.

### Q3: What is the difference between Weight-Only quantization (e.g. GPTQ/AWQ) and W8A8 (SmoothQuant)?
**Answer**:
- **Weight-Only (INT4/INT8)**: Quantizes stored weights to save VRAM. Weights are dequantized to FP16 on-the-fly during computation. Reduces VRAM by $4\times$--$8\times$; ideal for memory-bandwidth bound generation.
- **W8A8 (SmoothQuant)**: Quantizes both weights AND activations to INT8. Uses hardware INT8 Tensor Cores directly for matrix multiplication, achieving $2\times$ compute speedup in compute-bound prefill/batching workloads.

---

## 2. Capability Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands symmetric vs asymmetric quantization conceptually |
| **Competent** | Can implement `SymmetricQuantizer` and `AsymmetricQuantizer` scale & zero-point formulas |
| **Master** | Can implement block-wise INT4 weight quantization (`GroupQuantizer`) and build a `QuantizedLinear` layer |

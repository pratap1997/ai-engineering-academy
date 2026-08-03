# Module 011: Assessment & Readiness Check

## 1. Formative Questions

### Q1: Why GELU instead of ReLU in modern Transformers?
**Answer**: GELU is smooth and differentiable everywhere (ReLU has a kink at 0 causing dead neurons). GELU also has a probabilistic interpretation: it gates the input by the probability that a standard normal variable is ≤ x. Empirically, GELU outperforms ReLU in Transformer FFNs across all BERT/GPT experiments.

### Q2: What is the difference between LayerNorm and BatchNorm?
**Answer**: **BatchNorm** normalizes across the batch dimension per feature — requires large batch sizes and fails at batch_size=1. **LayerNorm** normalizes across the *feature* dimension per sample — works with any batch size, independent of sequence length, and stable for autoregressive generation (where batch=1 is common).

### Q3: How many parameters does a single Transformer Encoder block have?
**Answer** (for $d_\text{model}=512, H=8, d_{ff}=2048$):
- MHA: $4 \times 512^2 = 1{,}048{,}576$ (W_Q, W_K, W_V, W_O)
- FFN: $512 \times 2048 + 2048 + 2048 \times 512 + 512 = 2{,}099{,}712$
- LayerNorm × 2: $2 \times 2 \times 512 = 2{,}048$
- **Total per block ≈ 3.15M parameters**

---

## 2. Capability Rubric

| Level | Criteria |
|---|---|
| **Novice** | Can explain what FFN, LayerNorm, and residuals do conceptually |
| **Competent** | Can implement `TransformerEncoderBlock` and explain Pre-LN vs Post-LN |
| **Master** | Can stack a full `TransformerEncoder`, verify gradients numerically, calculate parameter counts per block, and name which production models use which design choices |

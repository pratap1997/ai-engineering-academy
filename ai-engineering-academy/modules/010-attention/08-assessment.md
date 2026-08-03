# Module 010: Assessment & Readiness Check

## 1. Readiness Check (Formative Questions)

### Q1: Why $\sqrt{d_k}$ Scaling?
**Question**: What happens to softmax outputs if we remove the $\sqrt{d_k}$ denominator for large $d_k$?
- **Answer**: For $d_k=512$, random dot products have std $\approx \sqrt{512} \approx 22.6$. Without scaling, softmax inputs are large, causing the softmax to saturate ($p \approx 1.0$ for the max, $\approx 0$ for all others), killing gradients and preventing effective learning.

### Q2: Self-Attention vs Cross-Attention
**Question**: What is the difference between Self-Attention and Cross-Attention?
- **Answer**: In **Self-Attention**, Q, K, and V all come from the same sequence (e.g. encoder attending to itself). In **Cross-Attention**, Q comes from the decoder's current state while K and V come from the encoder's output — allowing the decoder to "look at" relevant encoder positions.

### Q3: Multi-Head Attention Parameter Cost
**Question**: How many parameters does a Multi-Head Attention layer with $d_\text{model}=512$ and $H=8$ heads have?
- **Answer**: $4 \times d_\text{model}^2 = 4 \times 512^2 = 1{,}048{,}576 \approx 1M$ parameters ($W_Q + W_K + W_V + W_O$ each are $512 \times 512$).

---

## 2. Capability Evaluation Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands Q/K/V conceptually, but cannot write the attention score computation. |
| **Competent** | Can implement `ScaledDotProductAttention` and explain the $\sqrt{d_k}$ justification. |
| **Master** | Can implement `MultiHeadAttention` with head splitting/merging, causal masking, sinusoidal positional encoding, and verify gradients numerically. |

# Module 014: Advanced Positional Encodings (RoPE, ALiBi & Relative Position Encodings)

> "Traditional absolute positional encodings treat sequence positions as static tags added to input embeddings. Modern LLMs (LLaMA, Mistral, Falcon) replace them with relative rotation (RoPE) or linear distance biases (ALiBi), enabling superior length extrapolation and dynamic context extension."

---

## 1. Motivation: Beyond Absolute Positional Embeddings

In **Module 010**, we implemented Sinusoidal Absolute Positional Encodings ($\mathbf{x} + \mathbf{PE}$).
While effective for fixed sequence lengths ($T \le 512$), absolute positional encodings suffer from three core flaws:

1. **Poor Length Extrapolation**: A model trained on $T=2048$ fails when evaluated on $T=4096$ because position $2049$ has never been seen.
2. **Absolute vs Relative Misalignment**: Self-attention depends on *distance* between tokens ($m - n$), not absolute position $m$ or $n$. Adding absolute positions forces the model to learn relative relationships indirectly.
3. **Loss of Decay Property**: The attention score between token $m$ and token $n$ should naturally decrease as $|m - n|$ grows. Absolute embeddings do not guarantee this.

---

## 2. Three Modern Breakthroughs

| Technique | Used In | Mechanism |
|---|---|---|
| **Rotary Position Embedding (RoPE)** | LLaMA 1-3, Mistral, Qwen, PaLM | Rotates Query and Key vectors in 2D planes by angle proportional to position |
| **Attention with Linear Biases (ALiBi)** | BLOOM, MPT, Baichuan | Adds a static negative linear penalty proportional to distance directly to attention scores |
| **Relative Positional Encodings** | T5, Music Transformer | Adds a learned relative distance bias table to $QK^T$ |

---

## 3. Learning Outcomes

By completing this module, you will be able to:

1. **Implement Rotary Position Embeddings (RoPE)**: Rotate 2D query/key subvectors using complex rotation matrices or $\cos/\sin$ transformations.
2. **Implement ALiBi**: Generate per-head slope-weighted linear distance penalty matrices.
3. **Implement Relative Position Bias (T5-style)**: Bucket relative distances $(m - n)$ into a learned embedding lookup table.
4. **Demonstrate Length Extrapolation**: Show how RoPE and ALiBi maintain low perplexity when evaluating sequences longer than the training context.

---

## 4. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → Clock hands, distance penalties, and relative distance bucketing
03-mathematics.md       → RoPE complex rotation, ALiBi slope formulas, T5 relative bias
04-implementation.py    → RoPEEmbedding, ALiBiBias, RelativePositionBiasT5
05-experiments.py       → Dot product inner product decay vs distance, RoPE rotation properties
06-real-applications.md → LLaMA 3 RoPE implementation, YaRN length scaling, ALiBi context windowing
07-engineering-challenge.md → Build a RoPE-enhanced Multi-Head Attention forward pass
08-assessment.md        → Readiness check & self-assessment rubrics
09-references.md        → Su et al. (2021), Press et al. (2022), Shaw et al. (2018)
```

# Module 014: Assessment & Readiness Check

## 1. Formative Questions

### Q1: Why does RoPE preserve relative position information?
**Answer**: RoPE rotates 2D pairs of query vector $\mathbf{q}_m$ and key vector $\mathbf{k}_n$ by angles $m\theta$ and $n\theta$. The dot product $\langle \mathbf{R}_m \mathbf{q}, \mathbf{R}_n \mathbf{k} \rangle = \mathbf{q}^T \mathbf{R}_{n-m} \mathbf{k}$ depends exclusively on relative offset $n - m$.

### Q2: What is the main advantage of ALiBi over RoPE?
**Answer**: ALiBi requires zero complex number / trigonometric calculations and has absolute zero parameter overhead. Because it adds a fixed distance penalty directly to attention logits, it extrapolates seamlessly to sequence lengths significantly longer than those seen during training (e.g. train on 2K, test on 8K) without needing frequency scaling tuning like YaRN.

### Q3: Why does T5 use logarithmic distance bucketing?
**Answer**: Logarithmic bucketing reflects the linguistic intuition that precise relative position matters for nearby tokens (e.g., immediate previous word vs 2 words back), while long-range token distances (e.g., 500 tokens back vs 505 tokens back) carry almost identical positional context.

---

## 2. Capability Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands absolute vs relative positional encodings conceptually |
| **Competent** | Can implement `RoPEEmbedding` and `ALiBiBias` from scratch |
| **Master** | Can integrate RoPE into Multi-Head Attention, prove relative inner product invariance, and explain YaRN context scaling |

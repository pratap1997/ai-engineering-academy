# Module 010: Attention Mechanisms & Scaled Dot-Product Attention

> "Attention is the mechanism that allows neural networks to dynamically focus on the most relevant parts of the input for each output token — eliminating fixed bottleneck hidden states and enabling arbitrary long-range dependencies in O(1) steps."

---

## 1. Motivation: From Fixed Memory to Dynamic Focus

In **Modules 008–009**, RNNs and LSTMs compress an entire input sequence into a single fixed hidden vector $\mathbf{h}_T$ or cell state $\mathbf{C}_T$.

This creates an **information bottleneck**: in translation, answering "who did Alice meet?" requires the model to retrieve a specific word from step $t=3$ while generating output at step $t=50$. A fixed hidden vector cannot carry all information.

**Attention** solves this by:
1. Keeping **all** intermediate hidden states (not just the last one).
2. Computing a **dynamic weighted sum** of all states for each output step — attending more to relevant positions.

---

## 2. The Three Roles: Query, Key, Value

Attention is inspired by database retrieval:
- **Query ($\mathbf{Q}$)**: What am I looking for? (current decoder state or token)
- **Key ($\mathbf{K}$)**: What does each position offer? (all encoder hidden states)  
- **Value ($\mathbf{V}$)**: What information do I actually retrieve? (content vectors)

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

---

## 3. Learning Outcomes

By completing this module, you will be able to:

1. **Implement Scaled Dot-Product Attention**: Build `ScaledDotProductAttention` in pure Python & NumPy.
2. **Implement Multi-Head Attention**: Build `MultiHeadAttention` with $H$ parallel heads.
3. **Derive the $\sqrt{d_k}$ Scaling Factor**: Prove why scaling prevents softmax gradient saturation.
4. **Implement Sinusoidal Positional Encoding**: Inject sequence position information without recurrence.

---

## 4. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → The database retrieval analogy & attention heat maps
03-mathematics.md       → Attention score derivation, softmax saturation proof, multi-head math
04-implementation.py    → ScaledDotProductAttention, MultiHeadAttention, SinusoidalPositionalEncoding
05-experiments.py       → Attention weight heat map, sqrt(d_k) saturation demo, positional encoding patterns
06-real-applications.md → BERT (Bidirectional Encoder), GPT (Causal/Masked), PyTorch nn.MultiheadAttention
07-engineering-challenge.md → Custom self-attention forward pass & gradcheck verification
08-assessment.md        → Readiness check & self-assessment rubrics
09-references.md        → Vaswani et al. (2017) & Bahdanau et al. (2015) citations
```

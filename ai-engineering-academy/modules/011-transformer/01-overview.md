# Module 011: The Transformer Block

> "The Transformer Encoder Block is a deceptively simple stacking of three ideas: Multi-Head Attention (which module 010 built), a position-wise Feed-Forward Network (two linear layers + GELU), and a Residual + LayerNorm wrapper that makes deep stacks stable."

---

## 1. Motivation: From Attention to a Full Building Block

**Module 010** gave us `MultiHeadAttention` — the dynamic routing mechanism. But attention alone is not enough for a Transformer. We need:

1. **Position-wise FFN**: A per-token dense network that adds non-linear capacity *after* attention has mixed token information.
2. **Residual Connections**: `x + Sublayer(x)` — ensures gradient flow in deep stacks (Module 007 ResNet insight, now applied to sequences).
3. **Layer Normalization**: Normalize activations per-token across the feature dimension — stable training for very deep (12–96 layer) models.

One **Transformer Encoder Block** wraps all three:
```
x  → LayerNorm → MultiHeadAttention → + x  →  LayerNorm → FFN → + x
```
*(Pre-LN variant — more stable than original Post-LN)*

---

## 2. Learning Outcomes

1. **Implement `TransformerFFN`**: Position-wise FFN with `d_ff = 4 × d_model`.
2. **Implement `LayerNorm`**: Normalize each token's feature vector independently.
3. **Implement `TransformerEncoderBlock`**: Single Pre-LN block with MHA + FFN.
4. **Stack into `TransformerEncoder`**: $N$ identical stacked blocks.

---

## 3. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → The "highway + mixer" intuition & Pre-LN stability
03-mathematics.md       → FFN equations, LayerNorm derivation, residual gradient proofs
04-implementation.py    → TransformerFFN, LayerNorm, TransformerEncoderBlock, TransformerEncoder
05-experiments.py       → FFN expansion ratio, Pre-LN vs Post-LN gradient norm comparison
06-real-applications.md → BERT-base (12 blocks, d=768, H=12), GPT-2 (12 blocks, d=768)
07-engineering-challenge.md → Full Encoder Block forward pass with shapes and residual gradcheck
08-assessment.md        → Readiness check
09-references.md        → Vaswani (2017) & Ba (2016) LayerNorm citations
```

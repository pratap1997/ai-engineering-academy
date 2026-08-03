# Module 011: Mental Model — Highway + Mixer & Pre-LN Stability

## 1. The "Highway + Mixer" Intuition

Think of a Transformer block as doing two alternating operations on a sequence of token vectors:

**Step 1 — Multi-Head Attention: The Social Mixer**
- Every token *talks to every other token* and gathers relevant information.
- After this step, each token's vector has been updated with context from the rest of the sequence.
- The residual connection `x + Attention(x)` ensures the **original token information is never lost**.

**Step 2 — FFN: The Per-Token Thinker**
- Each token's updated vector is passed through **the same 2-layer MLP independently**.
- There is *no* communication between tokens here — it's purely per-position computation.
- This is where the model stores factual knowledge (proven empirically by mechanistic interpretability research).
- The residual connection `x + FFN(x)` again preserves the attention output.

```
Input Tokens      Attention Mixer      FFN Thinker
[cat]  ──────→   [cat+context] ──→    [refined_cat]
[sat]  ──────→   [sat+context] ──→    [refined_sat]
[on]   ──────→   [on+context]  ──→    [refined_on]
```

---

## 2. Pre-LN vs Post-LN: Why Pre-LN is More Stable

**Original (Post-LN)** from "Attention is All You Need":
```
x → Sublayer(x) → x + Sublayer(x) → LayerNorm
```

**Pre-LN** (modern standard, GPT-2, LLaMA):
```
x → LayerNorm → Sublayer(LN(x)) → x + Sublayer(LN(x))
```

**Why Pre-LN is better**: In Post-LN, the residual path passes through LayerNorm — this can cause large gradient variance at early training. In Pre-LN, the residual path $x$ bypasses LayerNorm entirely, keeping gradients more stable throughout training without needing warmup scheduling.

> 💡 **Rule of thumb**: GPT-2, PaLM, LLaMA all use Pre-LN. Always default to Pre-LN for new architectures.

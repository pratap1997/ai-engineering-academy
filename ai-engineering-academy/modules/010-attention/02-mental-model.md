# Module 010: Mental Model — Database Retrieval & Attention Heat Maps

## 1. The Database Retrieval Analogy

Think of attention as a **soft, differentiable database lookup**:

```
Traditional Database:        Attention Mechanism:
─────────────────────        ───────────────────
QUERY: "cat"                 QUERY vector Q
KEYS: ["dog", "cat", "bird"] KEY vectors K₁, K₂, K₃
VALUES: [img1, img2, img3]   VALUE vectors V₁, V₂, V₃

Exact match:                 Soft weighted match:
Returns img2 (100% match)    Returns 0.05*V₁ + 0.90*V₂ + 0.05*V₃
```

Attention returns a **blended value** — weighted by similarity between the query and all keys.

---

## 2. Why Scale by $\sqrt{d_k}$? The Softmax Saturation Problem

In high-dimensional spaces ($d_k = 512$), the dot product $\mathbf{q} \cdot \mathbf{k}$ grows with magnitude:
- For random vectors with unit variance, $\text{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k$
- Standard deviation grows as $\sqrt{d_k}$

Without scaling, dot products become **very large** (e.g. $512$), pushing the softmax into its saturated region where gradients → $0$:

$$\text{softmax}([100, 0.1, 0.2]) \approx [1.0, 0.0, 0.0] \quad ← \text{gradient death}$$

By dividing by $\sqrt{d_k}$:
$$\text{softmax}\!\left(\frac{[100, 0.1, 0.2]}{\sqrt{512}}\right) = \text{softmax}([4.4, 0.004, 0.009]) \quad ← \text{healthy distribution}$$

---

## 3. Multi-Head Attention — Parallel Semantic Subspaces

A single attention head sees the sequence through one "lens." **Multi-Head Attention** runs $H$ parallel attention operations, each with different learned $\mathbf{W}_Q^h, \mathbf{W}_K^h, \mathbf{W}_V^h$ projections:

- **Head 1**: Might learn to attend to syntactic dependencies ("subject-verb agreement")
- **Head 2**: Might learn to attend to semantic similarity ("cat ↔ kitten")
- **Head 3**: Might learn to attend to positional recency

The results are concatenated and projected: $\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) \mathbf{W}_O$

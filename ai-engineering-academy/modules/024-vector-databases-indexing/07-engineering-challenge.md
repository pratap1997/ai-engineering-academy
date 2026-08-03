# Module 024: Engineering Challenge — Hybrid Dense-Sparse Vector Retriever

## 1. Challenge Task

Construct a self-contained `HybridRetriever` in pure Python & NumPy that:
1. Combines **Dense Vector Search** (Cosine Similarity via `FlatIndex` or `IVFIndex`) with **Sparse Keyword Search** (BM25 / TF-IDF keyword overlap).
2. Implements **Reciprocal Rank Fusion (RRF)** to combine dense rank $R_\text{dense}$ and sparse rank $R_\text{sparse}$:
   $$\text{RRF Score}(d) = \frac{1}{60 + R_\text{dense}(d)} + \frac{1}{60 + R_\text{sparse}(d)}$$
3. Evaluates retrieval accuracy on mixed semantic and keyword queries.

---

## 2. Validation Criteria

1. RRF score calculation combines dense and sparse ranks correctly.
2. Hybrid retrieval achieves higher combined recall than dense search or sparse search alone.
3. Zero NaNs or indexing errors.

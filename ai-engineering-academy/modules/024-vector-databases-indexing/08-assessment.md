# Module 024: Assessment & Readiness Check

## 1. Formative Questions

### Q1: Why is exact brute-force vector search (Flat Index) impractical for large-scale production RAG?
**Answer**: Brute-force search computes distance between the query vector and every single vector in the database ($O(N \cdot d)$ complexity). For $N=10,000,000$ vectors of dimension $1536$, a single query requires 15 billion floating-point operations, taking hundreds of milliseconds per request and preventing high-throughput serving.

### Q2: How does IVF-Flat reduce search latency, and what controls the recall-speed tradeoff?
**Answer**: IVF-Flat partitions vector space into $K$ Voronoi clusters using K-Means. During query execution, it evaluates only vectors within the $n_\text{probe}$ nearest clusters. $n_\text{probe}$ controls the tradeoff: smaller $n_\text{probe}$ yields faster search but lower recall, while larger $n_\text{probe}$ increases recall toward $100\%$ at the cost of higher latency.

### Q3: Why does HNSW graph search achieve $O(\log N)$ logarithmic time complexity?
**Answer**: HNSW builds a multi-layer graph where upper layers contain sparse long-distance highway connections and lower layers contain dense local connections. Search begins at the top layer, rapidly narrowing down the search region in large geographic jumps, then steps down layers to locate exact nearest neighbors in $O(\log N)$ steps.

---

## 2. Capability Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands distance metrics (L2, Cosine) and the difference between Flat, IVF, and HNSW indexes |
| **Competent** | Can implement `FlatIndex`, `IVFIndex`, and measure Recall@K against exact ground truth |
| **Master** | Can build `HybridRetriever` with Reciprocal Rank Fusion (RRF), optimize $n_\text{probe}$ search parameters, and deploy production vector search |

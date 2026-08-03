# Module 024: Vector Databases & Dense Retrieval Indexing

> "Retrieval-Augmented Generation (RAG) and semantic search rely on retrieving relevant document embeddings from millions of high-dimensional vectors. Scanning every vector sequentially (Flat Index) has $O(N)$ linear complexity, taking seconds per query. Approximate Nearest Neighbor (ANN) indexing—such as IVF (Inverted File) and HNSW (Hierarchical Navigable Small World)—achieves sub-millisecond retrieval with $O(\log N)$ logarithmic complexity and $99\%$ recall!"

---

## 1. Motivation: The Search Complexity Wall

Searching a dataset of $N = 1,000,000$ vectors ($d = 1536$ dimensions):
- **Flat Index (Exact Brute-Force)**: Performs $1,000,000$ dot products. At $5\text{ ms}$ per query, throughput is capped at $200\text{ queries/sec}$.
- **IVF-Flat (Inverted File Index)**: Clusters vectors into $K = 1,000$ Voronoi centroids. Only searches the top $n_\text{probe} = 10$ nearest clusters ($10,000$ dot products). **$100\times$ faster!**
- **HNSW (Graph Index)**: Builds a multi-layer skip-list graph. Navigates from sparse top layers to dense bottom layers in $O(\log N)$ steps (**$500\times$ faster!**).

---

## 2. Vector Indexing Architecture Comparison

| Index Type | Search Time Complexity | Memory Overhead | Recall @ Top-10 | Construction Speed |
|---|---|---|---|---|
| **Flat Index** | $O(N \cdot d)$ | $0\%$ (Exact storage) | $100\%$ (Exact) | Instant ($O(1)$) |
| **IVF-Flat** | $O(\frac{n_\text{probe}}{K} \cdot N \cdot d)$ | Low ($K \cdot d$ centroids) | $95\text{--}98\%$ | Fast (K-Means) |
| **HNSW Graph** | $O(\log N \cdot d)$ | Medium (Graph edges $M$) | $\mathbf{98\text{--}99.9\%}$ | Medium (Graph insertion) |

---

## 3. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → The highway network & Voronoi neighborhood analogy
03-mathematics.md       → Cosine distance, K-Means Voronoi partitioning, HNSW layer probability
04-implementation.py    → FlatIndex, IVFIndex, HNSWIndex
05-experiments.py       → Recall vs QPS tradeoff & n_probe search accuracy sweep
06-real-applications.md → Milvus, Qdrant, Pinecone, FAISS production setups
07-engineering-challenge.md → Hybrid Dense-Sparse Vector Retrieval System
08-assessment.md        → Readiness check
09-references.md        → Malkov & Yashunin (2018), Jegou et al. (2011)
```

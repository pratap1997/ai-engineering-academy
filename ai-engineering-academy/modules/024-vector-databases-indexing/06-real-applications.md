# Module 024: Real Applications — Production Vector Databases & FAISS

## 1. Production Vector Databases (Qdrant / Milvus / Pinecone / FAISS)

In production RAG systems, vector indexing powers sub-millisecond retrieval across billions of embeddings:

```python
import faiss
import numpy as np

d = 1536  # OpenAI text-embedding-3-large dimension
n_list = 100

# 1. Instantiate IVF-PQ (Inverted File + Product Quantization) FAISS Index
quantizer = faiss.IndexFlatIP(d)  # Inner product / Cosine
index = faiss.IndexIVFPQ(quantizer, d, n_list, 16, 8)  # 16 sub-vectors, 8 bits

# 2. Train on dataset and add vectors
dataset_embeddings = np.random.randn(100000, d).astype(np.float32)
faiss.normalize_L2(dataset_embeddings)

index.train(dataset_embeddings)
index.add(dataset_embeddings)

# 3. Fast ANN search
index.nprobe = 10
query = np.random.randn(1, d).astype(np.float32)
faiss.normalize_L2(query)

distances, indices = index.search(query, k=5)
```

---

## 2. Quantization Types in Vector Databases

- **Scalar Quantization (SQ8)**: Quantizes FP32 embedding values into INT8 (4x RAM reduction).
- **Product Quantization (PQ)**: Splits $1536$-dim vector into 16 sub-vectors of 96 dimensions, replacing sub-vectors with codebook centroids ($32x$ RAM reduction).
- **Binary Quantization (BQ)**: Converts continuous embeddings into 1-bit binary vectors ($\text{sign}(x)$), enabling ultra-fast Hamming distance search on modern CPU AVX-512 instructions ($32x$ RAM reduction, $10x$ search speedup).

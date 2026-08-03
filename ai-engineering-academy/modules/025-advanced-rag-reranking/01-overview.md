# Module 025: Advanced RAG & Cross-Encoder Re-ranking

> "Basic RAG uses Bi-Encoders to independently map query and document chunks into a shared vector space. Because Bi-Encoders cannot perform token-level cross-attention between query words and document words, top-$K$ vector search often retrieves irrelevant chunks. Advanced RAG introduces **Cross-Encoder Re-ranking** and **Parent-Child Chunking**, boosting retrieval precision from $60\%$ to over $95\%$."

---

## 1. Motivation: Bi-Encoder vs Cross-Encoder Architecture

- **Bi-Encoder Retrieval (Fast, Low Precision)**:
  Query $q$ and Document $d$ are encoded **independently**:
  $$\text{Score}_{\text{Bi}} = \text{Cos}(\mathbf{E}(q), \mathbf{E}(d))$$
  - Computation is fast ($O(1)$ vector lookup via ANN index), but misses fine-grained word interactions.

- **Cross-Encoder Re-ranking (Slow, Extremely High Precision)**:
  Query $q$ and Document $d$ are concatenated into a **single input sequence**:
  $$\text{Input} = \text{[CLS]} \,\, q \,\, \text{[SEP]} \,\, d \,\, \text{[SEP]}$$
  - Full Transformer cross-attention is computed across all query and document tokens, capturing subtle negation, numerical relations, and exact semantic matches!

---

## 2. Two-Stage RAG Pipeline Architecture

```
                                [1,000,000 Document Chunks]
                                            │
                                            ▼
Stage 1: Bi-Encoder Vector Search ───> Retrieve Top-100 Chunks (Fast Recall Phase)
                                            │
                                            ▼
Stage 2: Cross-Encoder Re-ranking ───> Re-rank Top-10 Chunks (High Precision Phase)
                                            │
                                            ▼
                               [LLM Context Window Generation]
```

---

## 3. Parent-Child Chunking Strategy

Small chunks (e.g. 100 tokens) yield better vector embeddings for search, but lack full context for LLM generation.
Large chunks (e.g. 1000 tokens) preserve full context, but dilute vector embeddings during search.

**Parent-Child Solution**:
1. Split documents into small **Child Chunks** (100 tokens) for vector indexing.
2. Store mapping from each Child Chunk to its surrounding **Parent Chunk** (500 tokens).
3. When a Child Chunk is retrieved, feed the full **Parent Chunk** to the LLM context window!

---

## 4. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → The library index card & deep reading professor
03-mathematics.md       → Cross-attention score matrix & Softmax re-ranker loss
04-implementation.py    → ParentChildChunker, CrossEncoderReranker, AdvancedRAGPipeline
05-experiments.py       → MAP@10 comparison (Bi-Encoder vs Cross-Encoder) & Parent chunk expansion
06-real-applications.md → Cohere Rerank, BGE-Reranker-Large, LlamaIndex ParentDocumentRetriever
07-engineering-challenge.md → Complete End-to-End Advanced RAG Pipeline
08-assessment.md        → Readiness check
09-references.md        → Nogueira & Cho (2019), Gao et al. (2023)
```

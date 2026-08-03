# Module 025: Assessment & Readiness Check

## 1. Formative Questions

### Q1: Why does a Bi-Encoder vector search fail on subtle query-document relationships?
**Answer**: Bi-Encoders encode query and document into fixed vector representations independently. Because no cross-attention occurs between individual query tokens and document tokens, fine-grained semantic nuances (such as negation, numerical quantities, or exact phrase matches) are collapsed into the single vector, resulting in lower retrieval precision.

### Q2: What are the distinct roles of Stage 1 (Bi-Encoder) and Stage 2 (Cross-Encoder) in Two-Stage RAG?
**Answer**: Stage 1 (Bi-Encoder) prioritizes **high recall at low latency**, rapidly selecting top-100 candidates from millions of vectors using ANN search ($O(1)$ index lookup). Stage 2 (Cross-Encoder) prioritizes **high precision**, running full Transformer token cross-attention on the 100 candidate pairs to select the top-5 most relevant chunks for the LLM prompt.

### Q3: How does Parent-Child chunking resolve the chunk size dilemma?
**Answer**: Small chunks (e.g. 80 chars) create sharp, focused vector embeddings for search index matching, but lack surrounding narrative context. Large chunks (e.g. 500 chars) provide complete surrounding context to the LLM, but dilute vector search embeddings. Parent-Child chunking indexes small child chunks for vector search and retrieves their parent context chunks for LLM prompt generation!

---

## 2. Capability Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands the difference between Bi-Encoders and Cross-Encoders |
| **Competent** | Can implement `ParentChildChunker` and measure Precision@K and MAP@K |
| **Master** | Can build multi-query two-stage Advanced RAG pipelines with Cohere/BGE re-rankers in production |

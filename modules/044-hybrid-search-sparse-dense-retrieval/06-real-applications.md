# Real-World Applications of Hybrid Search

Hybrid search is the default architecture for modern high-accuracy retrieval systems.

## 1. Meilisearch Hybrid Search
Meilisearch v1.3+ introduced hybrid search. It uses BM25 for sparse retrieval and integrates with HNSW indexes for dense vectors. It blends results using an alpha parameter or Reciprocal Rank Fusion, ensuring typos are caught by semantic search while exact IDs are caught by keyword matching.

## 2. Elasticsearch & OpenSearch RAG
Enterprise search relies on Elasticsearch, which has added `text_expansion` and `knn` queries. Their `_search` endpoint allows combining a `match` query (BM25) with a `knn` query (dense) and fusing the scores mathematically to return highly precise context to LLMs.

## 3. Qdrant & Pinecone Hybrid Indexes
Vector database companies found that pure vectors struggle with keyword constraints. Both Pinecone (using SPLADE for sparse vectors) and Qdrant have native support for hybrid queries, storing both dense and sparse representations simultaneously.

## 4. E-commerce Search
When a user searches for "Samsung Galaxy S23 256GB Black", they want exact string matches. If they search for "good phone for taking pictures of the moon", they need semantic search. Hybrid search elegantly solves both seamlessly without manual query classification.

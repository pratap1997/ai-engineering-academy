# Module 025: Real Applications — Production Re-ranking & Cohere Rerank

## 1. Production Cohere Rerank & BGE-Reranker-Large Setup

In modern production RAG pipelines (LlamaIndex, LangChain, Haystack), two-stage retrieval is standard practice:

```python
# 1. Primary Vector Search (Bi-Encoder Stage)
vector_store = QdrantClient(url="http://localhost:6333")
query_embedding = openai_embeddings.embed_query("What is DPO alignment?")
candidate_nodes = vector_store.search(
    collection_name="ai_docs",
    query_vector=query_embedding,
    limit=50  # Wide top-50 recall candidate list
)

# 2. Secondary Cross-Encoder Re-ranking (Cross-Encoder Stage)
import cohere

co = cohere.Client(api_key="COHERE_API_KEY")
documents = [node.payload["text"] for node in candidate_nodes]

response = co.rerank(
    model="rerank-v3.5",
    query="What is DPO alignment?",
    documents=documents,
    top_n=5  # Narrow down to top-5 high precision context nodes
)

top_context_docs = [documents[r.index] for r in response.results]
```

---

## 2. Production Performance & Cost Profile

| Retrieval Stage | Latency | Cost per Query | Precision | Use Case |
|---|---|---|---|---|
| **Bi-Encoder Search** | $2\text{--}10\text{ ms}$ | $\$0.000001$ | $65\text{--}75\%$ | Retrieve top-100 candidates from millions of vectors |
| **Cross-Encoder Re-rank** | $30\text{--}80\text{ ms}$ | $\$0.0001$ | $\mathbf{92\text{--}98\%}$ | Re-rank top-100 down to top-5 for LLM prompt |

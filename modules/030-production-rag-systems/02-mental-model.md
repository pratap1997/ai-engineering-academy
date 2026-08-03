# 02 - Production RAG Systems: Mental Model

## The "Research Assistant" Mental Model
A prototype RAG system acts like a simplistic search engine: it finds matching keywords or embeddings and reads them aloud. A production RAG system acts like a **Research Assistant**. It doesn't just search; it:
1. **Understands** the intent behind the query.
2. **Finds** information from multiple sources using appropriate strategies.
3. **Synthesizes** the findings into a coherent, accurate answer.
4. **Checks** its own work before presenting it.

## Production RAG as a Manufacturing Pipeline
Think of production RAG as an assembly line:
- **Raw Material**: The user's query.
- **Classification Station**: Is this simple? Complex? Does it need routing?
- **Retrieval Station**: Fetching the parts (context) needed to build the answer.
- **Synthesis Station**: The LLM assembling the parts into a final product.
- **Quality Assurance**: Evaluating the product (RAGAS scores) before shipping.

## The Evaluation Loop
In manufacturing, QA isn't an afterthought. In RAG, RAGAS scores (Faithfulness, Relevancy, Precision) are your primary KPIs. If Faithfulness drops, your synthesis station is hallucinating. If Precision drops, your retrieval station is fetching junk.

## Caching Layers
Just as an assembly line stores commonly used parts nearby, RAG uses caching:
- **Query Cache (Exact Match)**: If a user asks the exact same string, return the saved answer instantly.
- **Embedding Cache (Semantic Match)**: If a query is semantically identical (e.g., "How do I reset my password?" vs. "Password reset steps"), return the cached answer.
- **Answer Cache**: Storing pre-computed answers for popular topics.

## The Stale Index Problem
An index is a snapshot of the world at time *T*. As the world moves to *T+1*, the index becomes stale. Like using an outdated map, a stale index leads the LLM to confidently provide wrong directions. Incremental updates and staleness metrics are essential to keep the map accurate.

## Pipeline Architecture

```text
User Query --> [ Query Cache ] --(hit)--> Answer
                     |
                  (miss)
                     v
             [ Query Classifier ]
                     |
            (Simple) / \ (Complex)
                    /   \
         [ BM25 ]       [ Vector Search + Reranker ]
               \         /
                \       /
                 v     v
           [ Context Assembly ]
                     |
                     v
               [ Generator (LLM) ]
                     |
                     v
          [ Evaluation (RAGAS) ]
                     |
                     v
                   Answer
```

## The Latency Budget
A 500ms response time is the gold standard for perceived real-time interaction. You must budget this time:
- Routing & Classification: 20ms
- Retrieval & Reranking: 180ms
- Generation (TTFT - Time To First Token): 300ms
Exceeding the budget in one area requires cutting corners in another (e.g., dropping reranking).

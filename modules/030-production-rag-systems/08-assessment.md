# 08 - Assessment

## Conceptual Questions
1. **Why is a prototype RAG system often insufficient for a production deployment?**
   *Answer*: Prototypes lack evaluation metrics, suffer from high latency, do not handle index staleness, and fail to gracefully handle out-of-distribution or complex queries.
2. **What does the 'Faithfulness' metric measure in the RAGAS framework?**
   *Answer*: It measures the hallucination rate by ensuring every claim in the generated answer can be directly attributed to the retrieved context.
3. **Explain the purpose of Query Routing in a production pipeline.**
   *Answer*: Not all queries require a full, expensive RAG pipeline. Routing allows simple queries to be answered from cache or skipped reranking, saving latency and compute costs.

## Math Questions
4. **Calculate the Faithfulness score**: An LLM generates an answer with 5 distinct claims. Upon evaluation, 4 of those claims are found in the retrieved context, and 1 is hallucinated.
   *Answer*: $4 / 5 = 0.8$ or 80%.
5. **Calculate the F1 Score**: A pipeline has a Context Precision of 0.6 and a Context Recall of 0.8.
   *Answer*: $2 \times (0.6 \times 0.8) / (0.6 + 0.8) = 0.96 / 1.4 \approx 0.685$.
6. **Calculate Staleness**: An index has 10,000 documents. Since the last build, 500 documents were updated and 200 were deleted.
   *Answer*: $700 / 10000 = 0.07$ or 7%.

## Implementation Questions
7. **Write the pseudocode for a two-level RAG cache.**
   *Answer*:
   ```python
   def get(query):
       if query in exact_cache: return exact_cache[query]
       q_emb = embed(query)
       for c_query, c_ans in semantic_cache:
           if cosine_sim(q_emb, embed(c_query)) > 0.95: return c_ans
       return None
   ```
8. **How would you implement a latency budget check that triggers a fallback?**
   *Answer*: Wrap the reranker in a timer. If `time_elapsed(retrieval) > budget_alloc`, skip the reranker and pass the unranked retrieved documents directly to the generator to save time.

## Judgment Questions
9. **You are building RAG for a medical diagnosis tool. Which RAGAS metric is the most critical to optimize, and why?**
   *Answer*: Faithfulness. A hallucinated answer (unfaithful to the medical context) could result in patient harm. Context Precision is secondary to ensuring that whatever is said is 100% grounded.
10. **Your latency budget is 500ms, but your LLM takes 400ms just to generate the first token (TTFT). What architectural changes should you make?**
   *Answer*: You must implement aggressive semantic caching to bypass the LLM entirely for common queries. For cache misses, you may need to switch to a smaller, faster LLM model or utilize streaming to improve perceived latency, while cutting the retrieval budget down to <100ms.

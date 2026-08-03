# 07 - Engineering Challenge

## Challenge: Build an Evaluated Production RAG System

Your task is to take the prototype pieces from previous modules and assemble them into a production-grade pipeline that can pass strict quality gates.

### Requirements

1. **Evaluation Gates**: Your pipeline must include a RAGAS evaluator. On the test set, it must achieve:
   - Faithfulness > 0.80
   - Answer Relevancy > 0.75
   - Context Precision > 0.70

2. **Query Routing**: Implement a `QueryClassifier` that identifies simple queries versus complex queries. Simple queries should skip the expensive reranking step to save latency.

3. **Adaptive Caching**: Implement a `RAGCache` that checks not just for exact string matches, but uses embedding similarity to serve cached answers for semantically identical questions.

4. **Latency Constraints**: The entire pipeline must execute in under 400ms end-to-end on average (use mocked retrievers and generators that sleep to simulate latency).

### Success Criterion
Run your pipeline against the provided 10 test cases in `05-experiments.py`. The output must show all 10 cases passing the RAGAS thresholds, with an average latency of <400ms.

### Rules
- Do NOT use external frameworks (LangChain, LlamaIndex, RAGAS). Build the logic from scratch to understand the math.
- Do NOT make actual API calls. Use deterministic mock embeddings and mocked LLM responses.

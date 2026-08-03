# Module 025: Engineering Challenge — Multi-Query Parent-Child Re-ranking RAG

## 1. Challenge Task

Construct a self-contained `MultiQueryAdvancedRAG` engine in pure Python & NumPy that:
1. Accepts a user query and generates $M=3$ **Query Rewrites / Variations** (simulating multi-query RAG expansion).
2. Performs Bi-Encoder retrieval across all $M$ query variations against a child chunk vector database.
3. De-duplicates child chunk candidates while accumulating candidate frequency scores.
4. Performs Cross-Encoder Re-ranking on the de-duplicated candidate set to pick the top-$K$ best parent context chunks.

---

## 2. Validation Criteria

1. Multi-query variation generation produces unique query strings.
2. Candidate de-duplication merges duplicate child chunk IDs correctly.
3. Top-$K$ output returns valid parent context text with zero NaNs.

# Assessment: Hybrid Search

**1. What is the primary weakness of pure Dense Vector Retrieval?**
- A) It is too slow.
- B) It struggles with exact keyword matching, IDs, and out-of-vocabulary terms.
- C) It requires too much memory.
- D) It cannot handle long documents.
**Answer:** B

**2. What does BM25 stand for?**
- A) Best Match 25
- B) Basic Metric 25
- C) Boolean Match 25
- D) Big Matrix 25
**Answer:** A

**3. In the Alpha-blending formula $S_{hybrid} = \alpha S_{dense} + (1-\alpha) S_{sparse}$, what happens when $\alpha = 0$?**
- A) The search is purely semantic.
- B) The search is purely exact keyword match (sparse).
- C) Both are weighted equally.
- D) The search fails.
**Answer:** B

**4. Why is normalization required before Alpha-blending BM25 and Cosine Similarity scores?**
- A) Because BM25 scores are unbounded while Cosine Similarity is typically bounded.
- B) Because Cosine Similarity is always negative.
- C) To make the code run faster.
- D) It is not required.
**Answer:** A

**5. What is Reciprocal Rank Fusion (RRF)?**
- A) A method of multiplying raw scores together.
- B) A fusion technique that combines results based on their relative ranks rather than absolute scores.
- C) A neural network layer for re-ranking.
- D) A database index structure.
**Answer:** B

**6. What is the role of the constant $k$ in RRF ($1 / (k + rank)$)?**
- A) It limits the total number of documents returned.
- B) It smoothes the score distribution, ensuring the top 1 result doesn't infinitely overpower the rank 2 result.
- C) It stands for the number of dimensions in the vectors.
- D) It determines the learning rate.
**Answer:** B

**7. Which scenario favors Sparse Retrieval (BM25)?**
- A) "Show me movies about a team pulling off a heist in a dream"
- B) "ERR_CONNECTION_RESET_104"
- C) "How do I fix my leaky faucet?"
- D) "What is the capital of France?"
**Answer:** B

**8. Which scenario favors Dense Retrieval?**
- A) Searching for a specific patient ID: "998-234-11"
- B) "Products similar to a red winter coat"
- C) Finding exact mentions of a CEO's name.
- D) SQL log grep.
**Answer:** B

**9. What is a common way to implement Sparse representations in vector databases without full BM25?**
- A) SPLADE (Sparse Lexical and Expansion Model)
- B) Word2Vec
- C) ResNet
- D) GPT-4
**Answer:** A

**10. What is a key advantage of RRF over Alpha-blending?**
- A) RRF requires no parameter tuning for the score distributions (score calibration).
- B) RRF is computationally faster.
- C) RRF uses less memory.
- D) RRF requires only one list.
**Answer:** A

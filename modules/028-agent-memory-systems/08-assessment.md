# 08 - Assessment

## Questions

**Conceptual**
1. Explain the difference between Working Memory and Episodic Memory in the context of an LLM agent.
2. What is the "session amnesia problem" and how does the Mem0 architecture solve it?
3. In human cognition, what is Procedural Memory, and how does it map to AI agents?

**Mathematical**
4. Given the Ebbinghaus forgetting equation $R = e^{-t/S}$, if a memory has a strength $S=2$, what is its retention $R$ after $t=4$ days?
5. How does increasing a memory's access count mathematically affect its retention over time in a spaced repetition system?
6. What is the theoretical capacity (in number of patterns) of a Hopfield Network with 100 neurons?

**Implementation**
7. Why is Cosine Similarity preferred over Euclidean distance when comparing dense text embeddings?
8. In the provided `MemoryStore` implementation, what triggers a memory to be deleted?

**Judgment**
9. You are building an AI doctor. Would you prioritize Recency or Importance when retrieving patient records? Why?
10. If your agent's vector database grows too large, what are two algorithmic strategies you can implement to reduce its size without losing critical information?

---

## Model Answers

1. **Working Memory** is the immediate context window (prompt) passed to the LLM; it is fast but strictly limited in size. **Episodic Memory** is the long-term log of past interactions, usually stored in a vector database and retrieved via similarity search.
2. The **session amnesia problem** is the fact that LLMs are stateless and forget previous API calls. Mem0 solves this by using an Extract-Update-Retrieve loop to persistently store user facts outside the context window.
3. **Procedural Memory** is the unconscious knowledge of *how* to do things. In AI agents, this maps to tool schemas, system prompts, and action libraries that define how the agent should behave.
4. $R = e^{-4/2} = e^{-2} \approx 0.135$.
5. Increasing access count increases the memory strength $S$. Since $S$ is in the denominator of the exponent ($e^{-t/S}$), a larger $S$ makes the negative exponent closer to zero, resulting in a higher retention $R$.
6. The capacity is approximately $0.14 \times N$. For 100 neurons, it is $0.14 \times 100 = 14$ patterns.
7. Cosine similarity measures the angle between vectors, making it invariant to the magnitude of the vectors. This is crucial for text embeddings where vector length might vary based on sentence length or word frequency, but the semantic "direction" remains the same.
8. A memory is deleted during `consolidate()` if its retention drops below 0.1, calculated using the forgetting curve based on elapsed time and memory strength.
9. **Importance**. While recent symptoms are relevant, long-term critical facts (like allergies, chronic conditions, or past surgeries) are vital and must not be forgotten or pushed out by trivial recent conversations.
10. (1) **Automatic Forgetting**: Apply a decay function and delete memories that fall below a threshold. (2) **Consolidation**: Merge highly similar semantic memories into a single, stronger memory.

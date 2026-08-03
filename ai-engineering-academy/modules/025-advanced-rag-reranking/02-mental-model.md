# Module 025: Mental Model — Library Index Cards & The Deep-Reading Professor

## 1. Bi-Encoder vs Cross-Encoder Mental Analogy

Think of document retrieval like a university library research task:

- **Bi-Encoder (The Library Card Catalog)**:
  - The librarian converts every book title into an index code independently.
  - When you ask a question, the librarian looks up matching index codes in seconds.
  - *Limitation*: The librarian only reads the summaries on the index cards, not the actual relationship between your exact question and every sentence in the book.

- **Cross-Encoder (The Subject Matter Expert Professor)**:
  - The professor sits down with your question and 20 candidate books.
  - The professor reads your question side-by-side with each book page, analyzing every subtle nuance, keyword relation, and context.
  - *Result*: The professor ranks the 5 most relevant pages with near $100\%$ precision!

---

## 2. Parent-Child Chunking Mental Model

- **Child Chunk (Search Snag)**: A tiny 100-token sentence fragment. It acts like a sharp hook that easily snags matching vector search queries.
- **Parent Chunk (Context Net)**: A full 500-token paragraph surrounding the sentence. Once the child hook catches a query, it pulls up the entire parent net, providing complete surrounding context to the LLM without hallucination!

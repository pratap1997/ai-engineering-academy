# Engineering Challenge: Spatial Layout-Aware Document RAG Ingestion Pipeline

## The Goal
Build a pipeline that ingests a simulated document containing text, headers, and a table. You must parse the table cells accurately, chunk the document based on its spatial layout, and successfully retrieve the exact table chunk when queried.

## Requirements
1. **Bounding Box Logic:** Implement an `Intersection over Union (IoU)` function to analyze overlapping regions.
2. **Table Parsing:** Given a list of cell dictionaries (with `row`, `col`, and `text`), reconstruct the table into a formatted string (e.g., CSV).
3. **Layout Chunking:** Process a list of `DocumentBlock` objects. Group contiguous text blocks, but isolate `table` and `image` blocks into their own distinct chunks.
4. **Retrieval:** Implement a frequency-based retrieval method that returns the most relevant chunk for a query.

## Constraints
- Do not use external ML frameworks (like PyTorch or TensorFlow) for this implementation.
- Use only standard Python libraries (`math`, `json`, `re`).
- All 16 unit tests provided in the module must pass.

Good luck!

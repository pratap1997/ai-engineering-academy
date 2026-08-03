# Engineering Challenge: Build a High-Precision Hybrid Search Retrieval Pipeline

## The Challenge
You have been hired by a medical documentation company. Doctors search the database using exact medical codes (e.g., "ICD-10-J45") and vague descriptions ("patient has trouble breathing at night").

Build a Hybrid Search pipeline that successfully retrieves the correct documents for BOTH query types.

## Requirements
1. **Implement `BM25Scorer`**: No cheating with libraries. Implement TF-IDF math.
2. **Implement `DenseVectorRetriever`**: Use cosine similarity for the mock vectors.
3. **Implement Alpha-Blending**: The alpha parameter should be configurable.
4. **Implement Reciprocal Rank Fusion (RRF)**: Implement RRF with k=60.
5. **No ML Frameworks**: Only standard Python and `math`.

## Validation
Your code must pass the provided 16 tests in `test_044_hybrid_search.py`. 
Run the tests using: `python -m pytest tests/`

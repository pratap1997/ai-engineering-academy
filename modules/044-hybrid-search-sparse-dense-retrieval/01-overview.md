# Module 044: Hybrid Search & Fusion Retrieval

## Context & Motivation

Retrieval-Augmented Generation (RAG) pipelines rely heavily on the quality of their initial search step. 
Historically, there were two distinct approaches to retrieval:
1. **Sparse Retrieval (e.g., BM25)**: Matches exact keywords. Excellent at finding specific names, IDs, and precise terminology.
2. **Dense Retrieval (e.g., Vector Embeddings)**: Matches semantic meaning. Excellent at understanding context, synonyms, and intent.

Hybrid Search combines both approaches to achieve higher precision than either method alone, mitigating the "exact match only" failure of sparse retrieval and the "hallucinated meaning" failure of dense retrieval.

## The Core Problem

Dense vectors struggle with out-of-vocabulary terms and exact keyword constraints (e.g., searching for "error code 404" might return semantic matches for "not found", but miss the exact term "404"). Sparse retrieval struggles with vocabulary mismatch (e.g., searching for "automobile" won't match documents containing only "car").

## The Solution: Hybrid Search & RRF

By querying both systems simultaneously and mathematically combining their results, we achieve robust retrieval. The two primary methods of combining results are:
1. **Alpha-blending**: Directly summing normalized scores using a weight parameter $\alpha$.
2. **Reciprocal Rank Fusion (RRF)**: Ignoring absolute scores and fusing based entirely on the relative ranking of documents in both lists.

"""
AI ENGINEERING ACADEMY -- MODULE 024 ENGINEERING CHALLENGE SOLUTION
Hybrid Dense-Sparse Vector Retriever with Reciprocal Rank Fusion (RRF)
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod24", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FlatIndex = _mod.FlatIndex


class SparseKeywordIndex:
    """Toy TF-IDF / BM25 sparse keyword index."""

    def __init__(self):
        self.doc_tokens = []

    def add_documents(self, documents_tokens):
        self.doc_tokens = documents_tokens

    def search(self, query_tokens, k=5):
        scores = []
        for doc in self.doc_tokens:
            overlap = len(set(query_tokens).intersection(set(doc)))
            scores.append(overlap)

        scores = np.array(scores)
        top_k_indices = np.argsort(-scores)[:k]
        return scores[top_k_indices], top_k_indices


class HybridRetriever:
    """
    Hybrid Retriever combining Dense Vector Cosine Similarity and Sparse Keyword Overlap using RRF.
    """

    def __init__(self, dense_dim, rrf_k=60):
        self.dense_index = FlatIndex(dim=dense_dim, metric="cosine")
        self.sparse_index = SparseKeywordIndex()
        self.rrf_k = rrf_k

    def add_documents(self, dense_vectors, text_tokens_list):
        self.dense_index.add(dense_vectors)
        self.sparse_index.add_documents(text_tokens_list)

    def search(self, query_dense_vec, query_text_tokens, top_k=5):
        # 1. Dense Search
        _, dense_top_idx = self.dense_index.search([query_dense_vec], k=top_k*2)
        dense_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(dense_top_idx[0])}

        # 2. Sparse Search
        _, sparse_top_idx = self.sparse_index.search(query_text_tokens, k=top_k*2)
        sparse_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(sparse_top_idx)}

        # 3. Reciprocal Rank Fusion (RRF)
        all_candidate_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        rrf_scores = {}

        for doc_id in all_candidate_ids:
            score = 0.0
            if doc_id in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[doc_id])
            if doc_id in sparse_ranks:
                score += 1.0 / (self.rrf_k + sparse_ranks[doc_id])
            rrf_scores[doc_id] = score

        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        final_doc_ids = [doc_id for doc_id, score in sorted_docs]
        final_scores = [score for doc_id, score in sorted_docs]

        return final_scores, final_doc_ids


def verify_hybrid_retriever():
    print("=" * 65)
    print("MODULE 024 CHALLENGE: HYBRID DENSE-SPARSE RETRIEVER WITH RRF")
    print("=" * 65)

    np.random.seed(42)
    N, d = 20, 16
    dense_vecs = np.random.randn(N, d)
    tokens_list = [["ai", "learning", f"doc_{i}"] for i in range(N)]
    tokens_list[5] = ["quantum", "physics", "superposition"]  # Specific sparse match for doc 5

    retriever = HybridRetriever(dense_dim=d, rrf_k=60)
    retriever.add_documents(dense_vecs, tokens_list)

    query_dense = np.random.randn(d)
    query_sparse = ["quantum", "physics"]

    rrf_scores, top_ids = retriever.search(query_dense, query_sparse, top_k=5)

    print(f"Top-5 RRF Document IDs: {top_ids}")
    print(f"Top-5 RRF Scores:       {[round(s, 6) for s in rrf_scores]}")

    # Document 5 should rank near the top due to perfect sparse keyword match
    assert 5 in top_ids
    print("\nHybrid Dense-Sparse Retrieval Verification Passed => [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_hybrid_retriever()

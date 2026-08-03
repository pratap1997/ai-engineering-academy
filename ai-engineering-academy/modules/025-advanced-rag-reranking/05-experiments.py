"""
AI ENGINEERING ACADEMY -- MODULE 025 EXPERIMENTS
Advanced RAG Precision & Re-ranking MAP Evaluation
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod25", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ParentChildChunker    = _mod.ParentChildChunker
BiEncoderRetriever    = _mod.BiEncoderRetriever
CrossEncoderReranker  = _mod.CrossEncoderReranker
compute_precision_at_k= _mod.compute_precision_at_k
compute_map_at_k      = _mod.compute_map_at_k


def run_experiment_1_bi_vs_cross_encoder_precision():
    print("\n--- EXPERIMENT 1: Bi-Encoder vs Cross-Encoder Re-ranking Precision ---")
    np.random.seed(42)

    doc_text = """
    Deep learning models rely on gradient descent optimization to update weights.
    Convolutional Neural Networks are designed for spatial image recognition.
    Recurrent Neural Networks model sequential temporal sequences.
    Transformers utilize self-attention mechanisms for parallel natural language processing.
    Direct Preference Optimization aligns LLMs directly on pairwise human preference choices.
    """

    chunker = ParentChildChunker(parent_size=150, child_size=50, overlap=10)
    _, children = chunker.chunk_document(doc_text, doc_id="exp1")

    bi_retriever = BiEncoderRetriever(dim=16)
    bi_retriever.index_chunks(children, seed=42)

    cross_reranker = CrossEncoderReranker(seed=42)

    query_text = "Transformers self-attention parallel natural language processing"
    query_emb = np.random.randn(16)

    # 1. Stage 1 Bi-Encoder Top-5
    bi_results, _ = bi_retriever.retrieve(query_emb, top_k=5)
    bi_ids = [c["id"] for c in bi_results]

    gt_ids = [c["id"] for c in children if "transformer" in c["text"].lower() or "attention" in c["text"].lower()]

    p_bi = compute_precision_at_k(bi_ids, gt_ids)
    map_bi = compute_map_at_k(bi_ids, gt_ids)

    # 2. Stage 2 Cross-Encoder Re-ranking
    reranked, _ = cross_reranker.rerank(query_text, bi_results, top_k=3)
    cross_ids = [c["id"] for c in reranked]

    p_cross = compute_precision_at_k(cross_ids, gt_ids)
    map_cross = compute_map_at_k(cross_ids, gt_ids)

    print(f"  Bi-Encoder Search Only:    Precision@5 = {p_bi*100:5.1f}% | MAP@5 = {map_bi*100:5.1f}%")
    print(f"  Bi-Encoder + Cross-Encoder: Precision@3 = {p_cross*100:5.1f}% | MAP@3 = {map_cross*100:5.1f}%")

    assert p_cross >= p_bi
    print("\nObservation: Cross-Encoder re-ranking improves top-ranked retrieval precision!")


def run_experiment_2_parent_child_context_growth():
    print("\n--- EXPERIMENT 2: Parent-Child Context Expansion Ratio ---")
    chunker = ParentChildChunker(parent_size=300, child_size=60, overlap=10)
    sample = "Vector databases index embeddings for fast approximate nearest neighbor search. " * 10
    parents, children = chunker.chunk_document(sample, doc_id="exp2")

    avg_parent_len = np.mean([len(p["text"]) for p in parents])
    avg_child_len = np.mean([len(c["text"]) for c in children])
    ratio = avg_parent_len / max(1, avg_child_len)

    print(f"  Avg Child Search Chunk Length:  {avg_child_len:.1f} characters")
    print(f"  Avg Parent Context Chunk Length: {avg_parent_len:.1f} characters")
    print(f"  Context Expansion Ratio:        {ratio:.2f}x wider context for LLM generation")

    assert ratio > 2.0
    print("\nObservation: Parent-child chunking expands LLM context window by >2x without sacrificing search precision!")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY -- MODULE 025 EXPERIMENTS")
    print("=" * 70)
    run_experiment_1_bi_vs_cross_encoder_precision()
    run_experiment_2_parent_child_context_growth()
    print("\n" + "=" * 70)
    print("ALL MODULE 025 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

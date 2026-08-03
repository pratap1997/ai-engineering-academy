"""
AI ENGINEERING ACADEMY -- MODULE 025 ENGINEERING CHALLENGE SOLUTION
Multi-Query Parent-Child Re-ranking RAG Engine
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod25", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ParentChildChunker   = _mod.ParentChildChunker
BiEncoderRetriever   = _mod.BiEncoderRetriever
CrossEncoderReranker = _mod.CrossEncoderReranker


class MultiQueryAdvancedRAG:
    """Multi-Query Expansion Two-Stage Parent-Child RAG Engine."""

    def __init__(self, dim=16):
        self.chunker = ParentChildChunker(parent_size=200, child_size=60, overlap=15)
        self.bi_encoder = BiEncoderRetriever(dim=dim)
        self.cross_encoder = CrossEncoderReranker()
        self.dim = dim

    def ingest_document(self, text, doc_id="doc_1"):
        parents, children = self.chunker.chunk_document(text, doc_id)
        self.bi_encoder.index_chunks(children, seed=42)
        return len(parents), len(children)

    def _generate_query_variations(self, base_query):
        """Simulates LLM multi-query rewrite generation."""
        return [
            base_query,
            f"Detailed explanation of {base_query}",
            f"Key mechanisms and concepts of {base_query}"
        ]

    def query(self, base_query, stage1_k=5, final_k=2, seed=42):
        np.random.seed(seed)
        variations = self._generate_query_variations(base_query)

        # 1. Retrieve candidates for each query variation
        candidates_map = {}
        for var in variations:
            q_emb = np.random.randn(self.dim)
            chunks, _ = self.bi_encoder.retrieve(q_emb, top_k=stage1_k)
            for chunk in chunks:
                candidates_map[chunk["id"]] = chunk

        unique_candidates = list(candidates_map.values())

        # 2. Cross-Encoder Re-rank candidates against original base query
        reranked_chunks, reranked_scores = self.cross_encoder.rerank(base_query, unique_candidates, top_k=final_k)

        # 3. Extract Parent Context Chunks
        parent_contexts = [c["parent_text"] for c in reranked_chunks]

        return parent_contexts, reranked_chunks, reranked_scores


def verify_multi_query_rag():
    print("=" * 65)
    print("MODULE 025 CHALLENGE: MULTI-QUERY PARENT-CHILD RAG ENGINE")
    print("=" * 65)

    sample_doc = """
    Quantization reduces LLM memory footprint by mapping 32-bit floating point weights into 8-bit or 4-bit integers.
    Symmetric quantization uses a zero scale offset, while asymmetric quantization includes a zero-point z.
    FlashAttention optimizes GPU memory traffic by tiling Query, Key, and Value blocks into SRAM caches.
    Speculative decoding uses a lightweight draft model to generate candidate token sequences verified in parallel by a target LLM.
    """

    engine = MultiQueryAdvancedRAG(dim=16)
    n_p, n_c = engine.ingest_document(sample_doc, doc_id="challenge_doc")

    query = "How does quantization map float weights to integers?"
    parent_contexts, chunks, scores = engine.query(query, stage1_k=4, final_k=2)

    print(f"Base Query: \"{query}\"")
    print(f"Parent Contexts Retrieved: {len(parent_contexts)}")

    for i, (score, chunk, ctx) in enumerate(zip(scores, chunks, parent_contexts)):
        print(f"  Rank {i+1} [Score={score:.4f}]: Child ID={chunk['id']}")
        print(f"    Context: \"{ctx[:85].strip()}...\"")

    assert len(parent_contexts) == 2
    assert scores[0] >= scores[1]
    print("\nMulti-Query Advanced RAG Verification Passed => [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_multi_query_rag()

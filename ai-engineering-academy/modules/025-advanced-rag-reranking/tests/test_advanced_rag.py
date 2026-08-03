"""
AI ENGINEERING ACADEMY -- MODULE 025 TEST SUITE
Comprehensive Pytest Suite for Advanced RAG & Re-ranking (16 Tests)
"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_mod25_dir = os.path.dirname(_dir)

_spec = importlib.util.spec_from_file_location("impl_mod25", os.path.join(_mod25_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ParentChildChunker     = _mod.ParentChildChunker
BiEncoderRetriever     = _mod.BiEncoderRetriever
CrossEncoderReranker   = _mod.CrossEncoderReranker
AdvancedRAGPipeline    = _mod.AdvancedRAGPipeline
compute_precision_at_k = _mod.compute_precision_at_k
compute_map_at_k       = _mod.compute_map_at_k

_spec_ch = importlib.util.spec_from_file_location("ch_mod25", os.path.join(_mod25_dir, "07-challenge-solution.py"))
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)
MultiQueryAdvancedRAG  = _mod_ch.MultiQueryAdvancedRAG
verify_multi_query_rag = _mod_ch.verify_multi_query_rag


# ===================================================================
# 1. PARENT-CHILD CHUNKER (4 tests)
# ===================================================================
class TestParentChildChunker:
    def test_chunk_document_creates_parents_and_children(self):
        chunker = ParentChildChunker(parent_size=100, child_size=30, overlap=5)
        text = "This is a long test document containing multiple sentences for chunking tests."
        parents, children = chunker.chunk_document(text, doc_id="d1")
        assert len(parents) > 0
        assert len(children) > len(parents)

    def test_child_contains_valid_parent_id(self):
        chunker = ParentChildChunker(parent_size=100, child_size=30, overlap=5)
        text = "Short text test for parent child mapping check."
        parents, children = chunker.chunk_document(text, doc_id="d2")
        parent_ids = {p["id"] for p in parents}
        for child in children:
            assert child["parent_id"] in parent_ids

    def test_child_parent_text_matches(self):
        chunker = ParentChildChunker(parent_size=100, child_size=30, overlap=5)
        parents, children = chunker.chunk_document("Sample sentence text.", doc_id="d3")
        for child in children:
            assert child["text"] in child["parent_text"]

    def test_empty_text_returns_empty_chunks(self):
        chunker = ParentChildChunker()
        parents, children = chunker.chunk_document("", doc_id="d4")
        assert len(parents) == 0
        assert len(children) == 0


# ===================================================================
# 2. BI-ENCODER & CROSS-ENCODER (4 tests)
# ===================================================================
class TestEncoders:
    def test_bi_encoder_retrieve_top_k_shape(self):
        bi = BiEncoderRetriever(dim=16)
        children = [{"id": f"c{i}", "text": f"text {i}"} for i in range(10)]
        bi.index_chunks(children)
        results, scores = bi.retrieve(np.random.randn(16), top_k=4)
        assert len(results) == 4
        assert len(scores) == 4

    def test_cross_encoder_score_pair_range(self):
        ce = CrossEncoderReranker()
        score = ce.score_pair("quantum computing", "quantum mechanics physics computing")
        assert 0.0 <= score <= 1.0

    def test_cross_encoder_rerank_orders_by_score(self):
        ce = CrossEncoderReranker()
        candidates = [
            {"id": "1", "text": "unrelated topic text"},
            {"id": "2", "text": "deep learning neural network optimization"}
        ]
        reranked, scores = ce.rerank("deep learning optimization", candidates, top_k=2)
        assert scores[0] >= scores[1]
        assert reranked[0]["id"] == "2"

    def test_bi_encoder_scores_sorted(self):
        bi = BiEncoderRetriever(dim=8)
        bi.index_chunks([{"id": f"c{i}", "text": f"text {i}"} for i in range(8)])
        _, scores = bi.retrieve(np.random.randn(8), top_k=5)
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))


# ===================================================================
# 3. METRICS & ADVANCED RAG PIPELINE (4 tests)
# ===================================================================
class TestMetricsPipeline:
    def test_precision_at_k(self):
        p = compute_precision_at_k(["a", "b", "c"], ["a", "c", "x"])
        np.testing.assert_allclose(p, 2.0 / 3.0)

    def test_map_at_k(self):
        m = compute_map_at_k(["a", "x", "b"], ["a", "b"])
        np.testing.assert_allclose(m, 5.0 / 6.0)

    def test_advanced_rag_pipeline_query(self):
        pipeline = AdvancedRAGPipeline(dim=16)
        pipeline.ingest_document("Neural networks are used for deep learning.", doc_id="d1")
        contexts, chunks, scores = pipeline.query("deep learning", np.random.randn(16), stage1_k=3, final_k=2)
        assert len(contexts) <= 2
        assert len(chunks) <= 2

    def test_advanced_rag_pipeline_no_nans(self):
        pipeline = AdvancedRAGPipeline(dim=16)
        pipeline.ingest_document("Test data string for non nan evaluation.", doc_id="d2")
        _, _, scores = pipeline.query("test data", np.random.randn(16), stage1_k=2, final_k=1)
        assert not np.isnan(scores).any()


# ===================================================================
# 4. MULTI-QUERY RAG & CHALLENGE (4 tests)
# ===================================================================
class TestMultiQueryChallenge:
    def test_challenge_verification_runs(self):
        verify_multi_query_rag()

    def test_multi_query_engine_returns_parent_contexts(self):
        engine = MultiQueryAdvancedRAG(dim=16)
        sample = """
        Quantization maps floating point values to integers.
        FlashAttention optimizes memory traffic by tiling Query Key and Value blocks into GPU SRAM.
        """
        engine.ingest_document(sample, doc_id="mq1")
        contexts, chunks, scores = engine.query("quantization memory", stage1_k=3, final_k=2)
        assert len(contexts) == 2

    def test_multi_query_variation_generation(self):
        engine = MultiQueryAdvancedRAG(dim=16)
        vars = engine._generate_query_variations("RAG search")
        assert len(vars) == 3
        assert vars[0] == "RAG search"

    def test_multi_query_engine_no_nans(self):
        engine = MultiQueryAdvancedRAG(dim=16)
        engine.ingest_document("Data input check for nan safety.", doc_id="mq2")
        _, _, scores = engine.query("nan safety", stage1_k=2, final_k=1)
        assert not np.isnan(scores).any()

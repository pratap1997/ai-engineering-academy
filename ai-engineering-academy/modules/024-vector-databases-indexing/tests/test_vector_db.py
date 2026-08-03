"""
AI ENGINEERING ACADEMY -- MODULE 024 TEST SUITE
Comprehensive Pytest Suite for Vector Databases & Indexing (16 Tests)
"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_mod24_dir = os.path.dirname(_dir)

_spec = importlib.util.spec_from_file_location("impl_mod24", os.path.join(_mod24_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FlatIndex           = _mod.FlatIndex
IVFIndex            = _mod.IVFIndex
HNSWIndex           = _mod.HNSWIndex
compute_recall_at_k = _mod.compute_recall_at_k
normalize_vectors   = _mod.normalize_vectors

_spec_ch = importlib.util.spec_from_file_location("ch_mod24", os.path.join(_mod24_dir, "07-challenge-solution.py"))
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)
HybridRetriever = _mod_ch.HybridRetriever
verify_hybrid_retriever = _mod_ch.verify_hybrid_retriever


# ===================================================================
# 1. FLAT INDEX (4 tests)
# ===================================================================
class TestFlatIndex:
    def test_flat_index_exact_cosine_match(self):
        idx = FlatIndex(dim=16, metric="cosine")
        v = np.eye(16)
        idx.add(v)
        dists, indices = idx.search([v[3]], k=1)
        assert indices[0][0] == 3
        np.testing.assert_allclose(dists[0][0], 0.0, atol=1e-5)

    def test_flat_index_l2_distance_metric(self):
        idx = FlatIndex(dim=16, metric="l2")
        v = np.random.randn(10, 16)
        idx.add(v)
        dists, indices = idx.search([v[0]], k=1)
        assert indices[0][0] == 0
        np.testing.assert_allclose(dists[0][0], 0.0, atol=1e-5)

    def test_flat_index_top_k_shape(self):
        idx = FlatIndex(dim=8, metric="cosine")
        idx.add(np.random.randn(20, 8))
        dists, indices = idx.search(np.random.randn(4, 8), k=5)
        assert indices.shape == (4, 5)

    def test_flat_index_no_nans(self):
        idx = FlatIndex(dim=8, metric="cosine")
        idx.add(np.random.randn(10, 8) * 10.0)
        dists, indices = idx.search(np.random.randn(2, 8), k=3)
        assert not np.isnan(dists).any()


# ===================================================================
# 2. IVF INDEX (4 tests)
# ===================================================================
class TestIVFIndex:
    def test_ivf_index_search_returns_k_items(self):
        ivf = IVFIndex(dim=16, n_list=5, n_probe=2)
        v = np.random.randn(50, 16)
        ivf.train_and_add(v, seed=42)
        dists, indices = ivf.search(np.random.randn(2, 16), k=5)
        assert len(indices[0]) == 5

    def test_ivf_high_n_probe_increases_recall(self):
        np.random.seed(42)
        N, d = 200, 16
        v = np.random.randn(N, d)
        q = np.random.randn(5, d)

        flat = FlatIndex(dim=d, metric="cosine")
        flat.add(v)
        _, gt_idx = flat.search(q, k=5)

        ivf_low = IVFIndex(dim=d, n_list=20, n_probe=1)
        ivf_low.train_and_add(v, seed=42)
        _, low_idx = ivf_low.search(q, k=5)

        ivf_high = IVFIndex(dim=d, n_list=20, n_probe=15)
        ivf_high.train_and_add(v, seed=42)
        _, high_idx = ivf_high.search(q, k=5)

        r_low = compute_recall_at_k(gt_idx, low_idx)
        r_high = compute_recall_at_k(gt_idx, high_idx)
        assert r_high >= r_low

    def test_ivf_centroids_shape(self):
        ivf = IVFIndex(dim=16, n_list=8, n_probe=2)
        ivf.train_and_add(np.random.randn(40, 16), seed=42)
        assert ivf.centroids.shape == (8, 16)

    def test_ivf_no_nans(self):
        ivf = IVFIndex(dim=16, n_list=5, n_probe=2)
        ivf.train_and_add(np.random.randn(30, 16), seed=42)
        dists, _ = ivf.search(np.random.randn(2, 16), k=3)
        assert not np.isnan(dists).any()


# ===================================================================
# 3. HNSW INDEX & RECALL UTILS (4 tests)
# ===================================================================
class TestHNSWIndex:
    def test_hnsw_search_shape(self):
        hnsw = HNSWIndex(dim=16, M=8)
        hnsw.build(np.random.randn(30, 16))
        dists, indices = hnsw.search(np.random.randn(3, 16), k=4)
        assert indices.shape == (3, 4)

    def test_recall_at_k_exact_match(self):
        gt = [[0, 1, 2], [3, 4, 5]]
        ann = [[0, 1, 2], [3, 4, 5]]
        recall = compute_recall_at_k(gt, ann)
        assert recall == 1.0

    def test_recall_at_k_zero_match(self):
        gt = [[0, 1, 2]]
        ann = [[3, 4, 5]]
        recall = compute_recall_at_k(gt, ann)
        assert recall == 0.0

    def test_normalize_vectors_unit_norm(self):
        v = np.random.randn(10, 16)
        v_norm = normalize_vectors(v)
        norms = np.linalg.norm(v_norm, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


# ===================================================================
# 4. HYBRID RETRIEVER & CHALLENGE (4 tests)
# ===================================================================
class TestHybridRetrieverChallenge:
    def test_challenge_verification_runs(self):
        verify_hybrid_retriever()

    def test_hybrid_retriever_rrf_scoring(self):
        retriever = HybridRetriever(dense_dim=8, rrf_k=60)
        vecs = np.random.randn(10, 8)
        tokens = [["cat"], ["dog"], ["bird"], ["cat", "dog"], ["fish"], ["cat"], ["dog"], ["bird"], ["cat"], ["dog"]]
        retriever.add_documents(vecs, tokens)

        scores, ids = retriever.search(vecs[0], ["cat"], top_k=3)
        assert len(ids) == 3
        assert scores[0] > 0.0

    def test_hybrid_retriever_returns_top_k_items(self):
        retriever = HybridRetriever(dense_dim=8, rrf_k=60)
        retriever.add_documents(np.random.randn(15, 8), [["a", "b"]] * 15)
        scores, ids = retriever.search(np.random.randn(8), ["a"], top_k=5)
        assert len(ids) == 5

    def test_hybrid_retriever_no_nans(self):
        retriever = HybridRetriever(dense_dim=8, rrf_k=60)
        retriever.add_documents(np.random.randn(10, 8), [["term"]] * 10)
        scores, _ = retriever.search(np.random.randn(8), ["term"], top_k=3)
        assert not np.isnan(scores).any()

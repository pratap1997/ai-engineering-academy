import pytest
import math
import sys
import importlib.util

# Dynamically import the implementation module
spec = importlib.util.spec_from_file_location("module_044_impl", "modules/044-hybrid-search-sparse-dense-retrieval/04-implementation.py")
module_044_impl = importlib.util.module_from_spec(spec)
sys.modules["module_044_impl"] = module_044_impl
spec.loader.exec_module(module_044_impl)

BM25Scorer = module_044_impl.BM25Scorer
DenseVectorRetriever = module_044_impl.DenseVectorRetriever
ReciprocalRankFusion = module_044_impl.ReciprocalRankFusion
HybridSearchEngine = module_044_impl.HybridSearchEngine

# --- CATEGORY 1: BM25 Calculation (4 Tests) ---

def test_bm25_initialization():
    scorer = BM25Scorer(k1=1.5, b=0.75)
    assert scorer.k1 == 1.5
    assert scorer.b == 0.75
    assert scorer.corpus_size == 0

def test_bm25_fit():
    corpus = [["hello", "world"], ["hello", "ai"], ["world", "of", "ai"]]
    scorer = BM25Scorer()
    scorer.fit(corpus)
    
    assert scorer.corpus_size == 3
    assert scorer.avgdl == 7.0 / 3.0
    assert "hello" in scorer.idf
    assert "world" in scorer.idf

def test_bm25_score_calculation():
    corpus = [["hello", "world"], ["hello", "ai"], ["world", "of", "ai"]]
    scorer = BM25Scorer()
    scorer.fit(corpus)
    
    # doc 0 has "hello", doc 1 has "hello"
    score_doc_0 = scorer.score(["hello"], 0)
    assert score_doc_0 > 0
    score_doc_2 = scorer.score(["hello"], 2)
    assert score_doc_2 == 0.0

def test_bm25_search_ordering():
    corpus = [["ai", "engineering"], ["ai", "ai", "ai"], ["hello", "world"]]
    scorer = BM25Scorer()
    scorer.fit(corpus)
    
    results = scorer.search(["ai"])
    assert len(results) == 3
    # doc 1 has more "ai" occurrences
    assert results[0][0] == 1 

# --- CATEGORY 2: Dense Vector Search (4 Tests) ---

def test_dense_retriever_initialization():
    retriever = DenseVectorRetriever()
    assert retriever.corpus_size == 0

def test_cosine_similarity():
    retriever = DenseVectorRetriever()
    v1 = [1.0, 0.0]
    v2 = [1.0, 0.0]
    assert math.isclose(retriever._cosine_similarity(v1, v2), 1.0)
    
    v3 = [0.0, 1.0]
    assert math.isclose(retriever._cosine_similarity(v1, v3), 0.0)

def test_dense_retriever_fit():
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]]
    retriever = DenseVectorRetriever()
    retriever.fit(vectors)
    assert retriever.corpus_size == 3

def test_dense_search_ordering():
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]]
    retriever = DenseVectorRetriever()
    retriever.fit(vectors)
    
    query = [1.0, 0.0]
    results = retriever.search(query)
    
    assert results[0][0] == 0
    assert results[1][0] == 2
    assert results[2][0] == 1

# --- CATEGORY 3: RRF Fusion Math (4 Tests) ---

def test_rrf_initialization():
    rrf = ReciprocalRankFusion(k=60)
    assert rrf.k == 60

def test_rrf_single_list():
    rrf = ReciprocalRankFusion(k=60)
    ranked_list = [(0, 0.9), (1, 0.8), (2, 0.7)]
    results = rrf.fuse([ranked_list])
    
    assert len(results) == 3
    assert results[0][0] == 0 # Rank 0 (score 1/61)
    assert results[1][0] == 1 # Rank 1 (score 1/62)

def test_rrf_multiple_lists():
    rrf = ReciprocalRankFusion(k=0) # simplified for easier math, k=0 means 1/(rank+1)
    list1 = [(0, 0.9), (1, 0.8)]
    list2 = [(1, 0.9), (2, 0.8)]
    
    results = rrf.fuse([list1, list2])
    # list1: 0 gets 1/1, 1 gets 1/2
    # list2: 1 gets 1/1, 2 gets 1/2
    # doc 0: 1
    # doc 1: 1.5
    # doc 2: 0.5
    assert results[0][0] == 1
    assert results[1][0] == 0
    assert results[2][0] == 2

def test_rrf_disjoint_lists():
    rrf = ReciprocalRankFusion(k=60)
    list1 = [(0, 0.9)]
    list2 = [(1, 0.9)]
    results = rrf.fuse([list1, list2])
    
    assert len(results) == 2
    assert results[0][1] == results[1][1]

# --- CATEGORY 4: Hybrid Pipeline (4 Tests) ---

def test_hybrid_engine_initialization():
    bm25 = BM25Scorer()
    dense = DenseVectorRetriever()
    engine = HybridSearchEngine(bm25, dense)
    assert engine.bm25_scorer == bm25
    assert engine.dense_retriever == dense

def test_hybrid_search_alpha_fusion():
    bm25 = BM25Scorer()
    dense = DenseVectorRetriever()
    
    corpus = [["apple"], ["banana"], ["apple", "banana"]]
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    
    bm25.fit(corpus)
    dense.fit(vectors)
    
    engine = HybridSearchEngine(bm25, dense)
    
    # Pure dense (alpha=1.0)
    results = engine.search_alpha_fusion(["apple"], [0.0, 1.0], alpha=1.0)
    assert results[0][0] == 1 # banana is [0.0, 1.0]

def test_hybrid_search_alpha_fusion_sparse():
    bm25 = BM25Scorer()
    dense = DenseVectorRetriever()
    
    corpus = [["apple"], ["banana"], ["apple", "banana"]]
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    
    bm25.fit(corpus)
    dense.fit(vectors)
    
    engine = HybridSearchEngine(bm25, dense)
    
    # Pure sparse (alpha=0.0)
    results = engine.search_alpha_fusion(["apple"], [0.0, 1.0], alpha=0.0)
    # doc 0 has "apple", doc 2 has "apple"
    assert results[0][0] in [0, 2]

def test_hybrid_search_rrf():
    bm25 = BM25Scorer()
    dense = DenseVectorRetriever()
    
    corpus = [["apple"], ["banana"], ["apple", "banana"]]
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    
    bm25.fit(corpus)
    dense.fit(vectors)
    
    engine = HybridSearchEngine(bm25, dense)
    results = engine.search_rrf(["apple"], [0.0, 1.0], k=60)
    
    assert len(results) == 3
    # Both systems contribute

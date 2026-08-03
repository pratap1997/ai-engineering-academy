import pytest
import sys
import os
import importlib.util

# Load implementation
module_dir = os.path.dirname(os.path.dirname(__file__))
impl_path = os.path.join(module_dir, "04-implementation.py")
spec = importlib.util.spec_from_file_location("module_030_impl", impl_path)
impl = importlib.util.module_from_spec(spec)
sys.modules["module_030_impl"] = impl
spec.loader.exec_module(impl)

# Category 1: Cache (4)
def test_rag_cache_stores_and_retrieves():
    cache = impl.RAGCache()
    res = impl.RAGResult(query=impl.RAGQuery("q", "simple"), answer="a", chunks=[], latency_ms=1, cache_hit=False, trace=[])
    cache.put("test query", res)
    assert cache.get("test query") == res

def test_rag_cache_hit_rate_tracking():
    cache = impl.RAGCache()
    res = impl.RAGResult(query=impl.RAGQuery("q", "simple"), answer="a", chunks=[], latency_ms=1, cache_hit=False, trace=[])
    cache.put("q1", res)
    cache.get("q1") # hit
    cache.get("q2_completely_different_length") # miss
    assert cache.hit_rate() == 0.5

def test_rag_cache_exact_match_hit():
    cache = impl.RAGCache()
    res = impl.RAGResult(query=impl.RAGQuery("q", "simple"), answer="a", chunks=[], latency_ms=1, cache_hit=False, trace=[])
    cache.put("exact", res)
    assert cache.get("exact") is not None
    assert cache.hits == 1

def test_rag_cache_lru_eviction():
    cache = impl.RAGCache(exact_capacity=2)
    res = impl.RAGResult(query=impl.RAGQuery("q", "simple"), answer="a", chunks=[], latency_ms=1, cache_hit=False, trace=[])
    cache.put("q1", res)
    cache.put("q2", res)
    cache.put("q3", res)
    assert len(cache.exact_cache) == 2

# Category 2: RAGAS Evaluation (4)
def test_faithfulness_score_perfect_alignment():
    evaluator = impl.RAGASEvaluator()
    # Mock logic uses word intersection
    assert evaluator.faithfulness("the sky is blue", ["the sky is blue"]) == 1.0

def test_faithfulness_score_no_alignment():
    evaluator = impl.RAGASEvaluator()
    assert evaluator.faithfulness("red car", ["the sky is blue"]) == 0.0

def test_context_precision_all_relevant():
    evaluator = impl.RAGASEvaluator()
    assert evaluator.context_precision("q", ["c1", "c2"], ["c1", "c2"]) == 1.0

def test_context_recall_full_retrieval():
    evaluator = impl.RAGASEvaluator()
    assert evaluator.context_recall(["c1", "c2"], ["c1", "c2"]) == 1.0

# Category 3: Pipeline Components (4)
def test_query_classifier_simple_query():
    classifier = impl.QueryClassifier()
    assert classifier.classify("What is Python?") == "simple"

def test_query_classifier_complex_query():
    classifier = impl.QueryClassifier()
    assert classifier.classify("Compare Python and Java") == "complex"

def test_latency_budget_within_bounds():
    budget = impl.LatencyBudget(500)
    budget.allocate("retrieval", 200)
    assert budget.check("retrieval", 150) == True

def test_latency_budget_exceeded_detection():
    budget = impl.LatencyBudget(500)
    budget.allocate("retrieval", 200)
    assert budget.check("retrieval", 250) == False

# Category 4: Index Management (4)
def test_index_manager_add_document():
    manager = impl.IndexManager()
    manager.add_document("doc1", "content")
    assert "doc1" in manager.documents
    assert manager.changes_since_rebuild == 1

def test_index_manager_delete_document():
    manager = impl.IndexManager()
    manager.add_document("doc1", "content")
    manager.delete_document("doc1")
    assert "doc1" not in manager.documents
    assert manager.changes_since_rebuild == 2

def test_index_manager_staleness_increases_after_update():
    manager = impl.IndexManager()
    manager.add_document("doc1", "content")
    manager.rebuild()
    assert manager.staleness_score() == 0.0
    manager.update_document("doc1", "new content")
    assert manager.staleness_score() == 1.0 # 1 change / 1 doc

def test_index_manager_rebuild_needed_threshold():
    manager = impl.IndexManager()
    manager.threshold = 0.5
    manager.add_document("doc1", "content")
    manager.add_document("doc2", "content")
    manager.rebuild()
    manager.update_document("doc1", "new")
    assert manager.rebuild_needed() == True # 1/2 >= 0.5

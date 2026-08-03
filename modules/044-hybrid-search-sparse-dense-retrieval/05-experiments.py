import time
from typing import List
import math

# Use relative import for the implementation module
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("module_044_impl", "modules/044-hybrid-search-sparse-dense-retrieval/04-implementation.py")
module_044_impl = importlib.util.module_from_spec(spec)
sys.modules["module_044_impl"] = module_044_impl
spec.loader.exec_module(module_044_impl)

HybridSearchEngine = module_044_impl.HybridSearchEngine
BM25Scorer = module_044_impl.BM25Scorer
DenseVectorRetriever = module_044_impl.DenseVectorRetriever

def run_experiments():
    print("Running Hybrid Search Experiments...")
    
    # 1. Setup Mock Data
    corpus = [
        ["the", "quick", "brown", "fox"],
        ["machine", "learning", "is", "fascinating"],
        ["hybrid", "search", "combines", "bm25", "and", "vectors"],
        ["vector", "search", "uses", "embeddings"]
    ]
    
    vectors = [
        [0.1, 0.9, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.8, 0.2, 0.1, 0.1]
    ]
    
    bm25 = BM25Scorer()
    bm25.fit(corpus)
    
    dense = DenseVectorRetriever()
    dense.fit(vectors)
    
    engine = HybridSearchEngine(bm25, dense)
    
    # Experiment 1: Alpha Weight Parameter Sweep
    print("\n--- Experiment 1: Alpha Sweep ---")
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    query_words = ["search", "vectors"]
    query_vector = [0.6, 0.4, 0.4, 0.4]
    
    for alpha in alphas:
        results = engine.search_alpha_fusion(query_words, query_vector, alpha=alpha)
        top_doc = results[0][0] if results else -1
        print(f"Alpha {alpha:.2f}: Top Document -> {top_doc}")

    # Experiment 2: RRF Constant k Tuning
    print("\n--- Experiment 2: RRF Constant Tuning ---")
    ks = [0, 10, 60, 100]
    for k in ks:
        results = engine.search_rrf(query_words, query_vector, k=k)
        print(f"RRF (k={k}): Top Document -> {results[0][0]} with RRF score {results[0][1]:.4f}")

if __name__ == "__main__":
    run_experiments()

from typing import List, Dict, Any
import importlib.util
import sys
import os

# Import implementation dynamically since filename starts with digit
spec = importlib.util.spec_from_file_location("implementation", os.path.join(os.path.dirname(__file__), "04-implementation.py"))
impl = importlib.util.module_from_spec(spec)
sys.modules["implementation"] = impl
spec.loader.exec_module(impl)

# Experiment 1: Cache Hit Rate
def run_experiment_1():
    print("--- Experiment 1: Cache Hit Rate ---")
    cache = impl.RAGCache(exact_capacity=2, semantic_capacity=5)
    
    # Mock some queries
    q1 = "How do I configure the server?"
    q2 = "Server configuration steps" # Similar enough in our mock embed
    q3 = "What is the meaning of life?"
    
    # Dummy result
    dummy_result = impl.RAGResult(
        query=impl.RAGQuery("dummy", "simple"),
        answer="Dummy answer",
        chunks=[], latency_ms=10, cache_hit=False, trace=[]
    )
    
    cache.put(q1, dummy_result)
    
    res1 = cache.get(q1) # Exact hit
    res2 = cache.get(q2) # Semantic hit (due to our length-based mock embedding)
    res3 = cache.get(q3) # Miss
    
    print(f"Hits: {cache.hits}, Misses: {cache.misses}")
    print(f"Hit Rate: {cache.hit_rate():.2f}")
    print()

# Experiment 2: RAGAS Evaluation
def run_experiment_2():
    print("--- Experiment 2: RAGAS Evaluation ---")
    evaluator = impl.RAGASEvaluator()
    
    question = "What color is the sky?"
    answer = "The sky is blue and clear."
    retrieved = ["The sky appears blue during the day.", "It is clear today."]
    ground_truth = ["The sky appears blue during the day."]
    
    scores = evaluator.full_evaluation(question, answer, retrieved, ground_truth)
    for k, v in scores.items():
        print(f"{k}: {v:.2f}")
    print()

# Experiment 3: Latency Budget
def run_experiment_3():
    print("--- Experiment 3: Latency Budget ---")
    budget = impl.LatencyBudget(total_ms=400.0)
    budget.allocate("retrieval", 150.0)
    budget.allocate("generation", 200.0)
    
    # Simulate runs
    print(f"Retrieval under budget? {budget.check('retrieval', 100.0)}")
    print(f"Generation under budget? {budget.check('generation', 250.0)}") # Fails
    
    report = budget.report()
    print(f"Total Budget: {report['total_budget']}ms")
    print(f"Actual Time: {report['total_actual']}ms")
    print(f"Within overall budget? {report['within_budget']}")
    print()

# Experiment 4: Query Classification
def run_experiment_4():
    print("--- Experiment 4: Query Classification ---")
    classifier = impl.QueryClassifier()
    
    queries = [
        "What is the capital of France?",
        "Compare the capital of France with the capital of Germany and explain the differences in population."
    ]
    
    for q in queries:
        cls = classifier.classify(q)
        print(f"Query: '{q[:30]}...' -> Class: {cls}")
    print()

if __name__ == "__main__":
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()

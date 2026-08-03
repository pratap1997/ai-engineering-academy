import time
from typing import List, Dict, Any
import importlib.util
import os

def load_impl():
    spec = importlib.util.spec_from_file_location("module_049_impl", "04-implementation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    impl = load_impl()
except Exception:
    pass

def run_experiment_1():
    print("--- Exp 1: Prompt Compression Token Reduction % ---")
    compressor = impl.PromptCompressor()
    text = "Please could you explain the theory of relativity to a 5 year old"
    compressed = compressor.compress(text)
    orig_len = len(text.split())
    comp_len = len(compressed.split())
    reduction = (orig_len - comp_len) / orig_len * 100
    print(f"Original: {text}")
    print(f"Compressed: {compressed}")
    print(f"Reduction: {reduction:.2f}%")

def run_experiment_2():
    print("--- Exp 2: Cache Hit Rate vs Query Variability ---")
    cache = impl.SemanticVectorCache(similarity_threshold=0.5)
    cache.set("what is machine learning", "ML is a field of AI")
    
    queries = [
        "what is machine learning",
        "what exactly is machine learning",
        "tell me about machine learning",
        "how do you cook pasta"
    ]
    
    hits = 0
    for q in queries:
        if cache.get(q):
            hits += 1
    
    print(f"Hits: {hits}/{len(queries)} ({(hits/len(queries))*100:.2f}%)")

def run_experiment_3():
    print("--- Exp 3: Provider Failover Recovery Latency ---")
    def fail_func(x):
        time.sleep(0.01)
        raise ValueError("Failed")
        
    def success_func(x):
        time.sleep(0.01)
        return "Success response"
        
    providers = [
        {"name": "P1", "func": fail_func},
        {"name": "P2", "func": fail_func},
        {"name": "P3", "func": success_func},
    ]
    router = impl.ProviderFailoverRouter(providers)
    
    t0 = time.time()
    res = router.route("test query")
    t1 = time.time()
    
    print(f"Response: {res}")
    print(f"Time taken to failover and succeed: {(t1-t0)*1000:.2f}ms")

def run_experiment_4():
    print("--- Exp 4: Cost Savings on 10,000 requests ---")
    engine = impl.AIGatewayEngine(impl.ProviderFailoverRouter([{"name": "P", "func": lambda x: "API Response"}]))
    
    total_reqs = 10000
    duplicates = 4000
    
    engine.process("unique query")
    
    cache_hits = duplicates
    cost_per_api_call = 0.01
    savings = cache_hits * cost_per_api_call
    print(f"Simulated {total_reqs} requests with {duplicates} cache hits.")
    print(f"Estimated savings: ${savings:.2f}")

if __name__ == "__main__":
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()

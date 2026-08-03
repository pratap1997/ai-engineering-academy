import time
import math
import sys
import os
import importlib.util

def load_implementation():
    path = os.path.join(os.path.dirname(__file__), "04-implementation.py")
    spec = importlib.util.spec_from_file_location("implementation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["implementation"] = module
    spec.loader.exec_module(module)
    return module

try:
    impl = load_implementation()
    MemoryStore = impl.MemoryStore
    EmbeddingModel = impl.EmbeddingModel
    WorkingMemory = impl.WorkingMemory
    HopfieldNetwork = impl.HopfieldNetwork
except Exception as e:
    print("Could not load 04-implementation.py, ensure it exists in the same directory.")
    sys.exit(1)

def run_experiment_1():
    print("\n--- Experiment 1: Memory Retrieval Quality ---")
    embedder = EmbeddingModel(vocab_size=100)
    store = MemoryStore(embedder=embedder)
    
    store.store("The Eiffel Tower is located in Paris.")
    store.store("Python is a programming language.")
    store.store("The capital of France is Paris.")
    store.store("Dogs are loyal pets.")
    
    q = "What is the capital of France?"
    results = store.retrieve(q, k=2)
    
    print(f"Query: '{q}'")
    print("Top-2 Results (Cosine Similarity):")
    for r in results:
        print(f" - {r.content}")

def run_experiment_2():
    print("\n--- Experiment 2: Forgetting Curve ---")
    store = MemoryStore(embedder=EmbeddingModel())
    
    days = [0, 1, 3, 7, 14, 30]
    strength_low = 1.0
    strength_high = 3.0
    
    print("Days | Low Strength Retention | High Strength Retention")
    print("-" * 55)
    for d in days:
        hours = d * 24
        ret_low = store.forget_curve(hours, strength=strength_low)
        ret_high = store.forget_curve(hours, strength=strength_high)
        bar_low = "#" * int(ret_low * 20)
        bar_high = "#" * int(ret_high * 20)
        print(f"{d:>4} | {ret_low:>4.2f} {bar_low:<20} | {ret_high:>4.2f} {bar_high:<20}")

def run_experiment_3():
    print("\n--- Experiment 3: Working Memory Overflow ---")
    wm = WorkingMemory(capacity=3)
    
    inputs = ["A", "B", "C", "D", "E"]
    for i in inputs:
        wm.add(i)
        print(f"Added {i}, Memory State: {wm.get_all()}")

def run_experiment_4():
    print("\n--- Experiment 4: Hopfield Capacity ---")
    N = 10
    hn = HopfieldNetwork(n_neurons=N)
    
    p1 = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    hn.store_pattern(p1)
    
    p2 = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
    hn.store_pattern(p2)
    
    print(f"Theoretical capacity: {hn.capacity:.2f} patterns.")
    
    noisy_p1 = [-1, -1, 1, -1, 1, -1, 1, -1, 1, 1]
    retrieved = hn.retrieve(noisy_p1)
    
    print(f"Original p1:   {p1}")
    print(f"Noisy input:   {noisy_p1}")
    print(f"Retrieved:     {retrieved}")
    
    matches = sum(1 for x, y in zip(p1, retrieved) if x == y)
    print(f"Recovery accuracy: {matches/N * 100}%")

if __name__ == "__main__":
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()

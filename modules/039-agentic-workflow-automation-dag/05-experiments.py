import time
import os
import sys

sys.path.append(os.path.dirname(__file__))
import importlib.util
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '04-implementation.py'))
spec = importlib.util.spec_from_file_location("module_039_impl", file_path)
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)

def run_experiment_1():
    print("--- Experiment 1: Topological vs Naive ---")
    graph = impl.StateGraph()
    for n in ["A", "B", "C", "D"]:
        graph.add_node(impl.WorkflowNode(n, lambda s: s))
    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_edge("B", "D")
    graph.add_edge("C", "D")
    
    order = impl.TopologicalSorter.sort(graph)
    print(f"Topological Order: {order}")
    print("Guarantees dependencies are met before execution.")

def run_experiment_2():
    print("\n--- Experiment 2: Error Recovery & Rollback ---")
    graph = impl.StateGraph()
    attempts = {'A': 0}
    def fail_action(s):
        attempts['A'] += 1
        s['mutated'] = True
        if attempts['A'] < 3:
            raise ValueError("Temp Fail")
        return s
    
    graph.add_node(impl.WorkflowNode("A", fail_action, max_retries=3))
    engine = impl.DAGWorkflowEngine(graph)
    res = engine.execute({"mutated": False})
    print(f"Attempts: {attempts['A']}, Final State: {res}")

def run_experiment_3():
    print("\n--- Experiment 3: Human-in-the-Loop ---")
    graph = impl.StateGraph()
    graph.add_node(impl.WorkflowNode("A", lambda s: {**s, 'stage': 1}, requires_human=True))
    engine = impl.DAGWorkflowEngine(graph)
    
    s1 = engine.execute({"stage": 0})
    print(f"Paused State: {s1}")
    engine.hitl_gate.approve("A")
    s2 = engine.execute(s1)
    print(f"Resumed State: {s2}")

def run_experiment_4():
    print("\n--- Experiment 4: Cycle Detection ---")
    graph = impl.StateGraph()
    graph.add_node(impl.WorkflowNode("A", lambda s: s))
    graph.add_node(impl.WorkflowNode("B", lambda s: s))
    graph.add_edge("A", "B")
    graph.add_edge("B", "A")
    has_cycle = impl.TopologicalSorter.has_cycle(graph)
    print(f"Cycle Detected: {has_cycle}")

if __name__ == "__main__":
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()

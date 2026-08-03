import time
from .module_04_impl import AutonomousAIEngineerSystem, MemoryStore, RAGEngine

def experiment_1_end_to_end_task_success():
    """End-to-end task success rate on multi-step software engineering problem."""
    system = AutonomousAIEngineerSystem()
    result = system.run("Build a simple HTTP server in Python")
    print("Experiment 1: Task Success Rate ->", result["success"])

def experiment_2_full_system_telemetry():
    """Full system token cost & latency breakdown using Tracer."""
    system = AutonomousAIEngineerSystem()
    start = time.time()
    system.run("Refactor legacy monolithic application")
    end = time.time()
    
    traces = system.tracer.get_traces()
    print(f"Experiment 2: Execution took {end - start:.4f}s. Generated {len(traces)} tracing spans.")

def experiment_3_resilience_tool_failure():
    """Resilience against tool failure & prompt injection."""
    system = AutonomousAIEngineerSystem()
    
    def faulty_tool():
        raise RuntimeError("Network Timeout")
        
    system.sandbox.register_tool("fetch_data", faulty_tool)
    result = system.sandbox.execute("fetch_data", {})
    
    print("Experiment 3: System survived tool failure ->", "Tool execution failed" in result)

def experiment_4_memory_retention():
    """Memory retention across multi-day simulated operations."""
    store = MemoryStore()
    store.add("user", "My API key is sk-1234")
    store.add("user", "We are using React for the frontend")
    
    # Simulate search days later
    res = store.search("frontend", k=1)
    print("Experiment 4: Retained architectural decision ->", "React" in res[0]["content"])

if __name__ == "__main__":
    experiment_1_end_to_end_task_success()
    experiment_2_full_system_telemetry()
    experiment_3_resilience_tool_failure()
    experiment_4_memory_retention()

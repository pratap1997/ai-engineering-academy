import os
import sys
import importlib.util
import time

def load_module():
    module_name = "module_040_impl"
    current_dir = os.path.dirname(__file__)
    file_path = os.path.abspath(os.path.join(current_dir, "04-implementation.py"))
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

mod = load_module()
Tracer = mod.Tracer
TokenCounter = mod.TokenCounter
CostCalculator = mod.CostCalculator
AgentProfiler = mod.AgentProfiler

def run_experiment_1_trace_overhead():
    print("Experiment 1: Trace Tree Construction Latency Overhead")
    tracer = Tracer()
    
    start = time.time()
    for _ in range(1000):
        tracer.start_trace()
        span1 = tracer.start_span("root")
        span2 = tracer.start_span("child1")
        tracer.end_span(span2)
        span3 = tracer.start_span("child2")
        tracer.end_span(span3)
        tracer.end_span(span1)
    
    duration = time.time() - start
    print(f"Overhead per trace (ms): {(duration / 1000) * 1000:.4f}")

def run_experiment_2_token_cost():
    print("\nExperiment 2: Token Cost Breakdown")
    tracer = Tracer()
    tracer.start_trace()
    
    # Simulate LLM call
    span = tracer.start_span("LLM Call")
    prompt = "What is the capital of France?"
    completion = "The capital of France is Paris."
    
    p_tokens = TokenCounter.count_tokens(prompt)
    c_tokens = TokenCounter.count_tokens(completion)
    cost = CostCalculator.calculate_cost("gpt-4", p_tokens, c_tokens)
    
    span.set_attribute("llm.usage.prompt_tokens", p_tokens)
    span.set_attribute("llm.usage.completion_tokens", c_tokens)
    span.set_attribute("llm.usage.total_tokens", p_tokens + c_tokens)
    span.set_attribute("llm.cost", cost)
    tracer.end_span(span)
    
    print(f"Prompt tokens: {p_tokens}")
    print(f"Completion tokens: {c_tokens}")
    print(f"Total Cost: ${cost:.6f}")

def run_experiment_3_p95_latency():
    print("\nExperiment 3: P95/P99 Latency Distribution")
    profiler = AgentProfiler()
    
    # Simulate 100 runs with varying latencies
    import random
    random.seed(42)
    for _ in range(100):
        tracer = Tracer()
        tracer.start_trace()
        span = tracer.start_span("agent_run")
        time.sleep(random.uniform(0.001, 0.01)) # Mock workload
        tracer.end_span(span)
        profiler.record_run(tracer.current_trace)
        
    print(f"P95 Latency (s): {profiler.get_p95_latency():.4f}")

def run_experiment_4_anomaly_detection():
    print("\nExperiment 4: Anomaly & Error Span Detection")
    tracer = Tracer()
    tracer.start_trace()
    span = tracer.start_span("Database_Query")
    try:
        raise ValueError("Connection Timeout")
    except Exception as e:
        span.add_event("exception", {"message": str(e)})
        tracer.end_span(span, status="ERROR")
    
    print(f"Span Status: {span.status}")
    print(f"Span Events: {span.events}")

if __name__ == "__main__":
    run_experiment_1_trace_overhead()
    run_experiment_2_token_cost()
    run_experiment_3_p95_latency()
    run_experiment_4_anomaly_detection()

import os
import sys
import importlib.util
import json
import pytest

def load_module():
    module_name = "module_040_impl"
    # The test file is in modules/040-production-agent-observability-tracing/tests/
    # The implementation is in modules/040-production-agent-observability-tracing/04-implementation.py
    current_dir = os.path.dirname(__file__)
    file_path = os.path.abspath(os.path.join(current_dir, "..", "04-implementation.py"))
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

mod = load_module()

# --- Span & Trace Hierarchy (4) ---

def test_span_initialization():
    span = mod.Span("TestSpan")
    assert span.name == "TestSpan"
    assert span.parent_id is None
    assert span.status == "OK"
    assert span.end_time is None

def test_trace_hierarchy():
    tracer = mod.Tracer()
    trace_id = tracer.start_trace()
    assert trace_id is not None
    root = tracer.start_span("Root")
    child = tracer.start_span("Child")
    assert child.parent_id == root.span_id
    tracer.end_span(child)
    tracer.end_span(root)
    assert len(tracer.current_trace.spans) == 2

def test_span_attributes():
    span = mod.Span("Test")
    span.set_attribute("key", "value")
    assert span.attributes["key"] == "value"

def test_span_events():
    span = mod.Span("Test")
    span.add_event("Error", {"msg": "Failed"})
    assert len(span.events) == 1
    assert span.events[0]["name"] == "Error"
    assert span.events[0]["attributes"]["msg"] == "Failed"


# --- Token & Cost Accounting (4) ---

def test_token_counter():
    text = "Hello world!"
    tokens = mod.TokenCounter.count_tokens(text)
    assert tokens == 3  # 12 chars // 4 = 3

def test_cost_calculation_gpt4():
    cost = mod.CostCalculator.calculate_cost("gpt-4", 1000, 1000)
    assert abs(cost - 0.09) < 1e-6  # 0.03 + 0.06

def test_cost_calculation_gpt35():
    cost = mod.CostCalculator.calculate_cost("gpt-3.5", 1000, 1000)
    assert abs(cost - 0.0035) < 1e-6 # 0.0015 + 0.002

def test_cost_calculation_unknown_model():
    cost = mod.CostCalculator.calculate_cost("unknown", 1000, 1000)
    assert cost == 0.0


# --- Quantile Latency Calculation (4) ---

def test_profiler_record_run():
    profiler = mod.AgentProfiler()
    trace = mod.Trace("t1")
    span = mod.Span("Root")
    span.end()
    trace.add_span(span)
    profiler.record_run(trace)
    assert profiler.total_runs == 1
    assert len(profiler.latencies) == 1

def test_profiler_metrics_aggregation():
    profiler = mod.AgentProfiler()
    trace = mod.Trace("t1")
    span = mod.Span("LLM")
    span.set_attribute("llm.usage.total_tokens", 50)
    span.set_attribute("llm.cost", 0.1)
    span.end("ERROR")
    trace.add_span(span)
    profiler.record_run(trace)
    assert profiler.errors == 1
    assert profiler.total_tokens == 50
    assert profiler.total_cost == 0.1

def test_p95_latency_empty():
    profiler = mod.AgentProfiler()
    assert profiler.get_p95_latency() == 0.0

def test_p95_latency_calculation():
    profiler = mod.AgentProfiler()
    for i in range(1, 101):
        trace = mod.Trace(f"t{i}")
        span = mod.Span("Root")
        span.start_time = 0
        span.end_time = i
        trace.add_span(span)
        profiler.record_run(trace)
    p95 = profiler.get_p95_latency()
    assert p95 == 96.0  # 95th index in 100 elements sorted 1-100 is 96 (index 95)


# --- Exporter & Profiler (4) ---

def test_exporter_json_format():
    trace = mod.Trace("t1")
    span = mod.Span("Root")
    span.end()
    trace.add_span(span)
    json_str = mod.SpanExporter.export_to_json(trace)
    data = json.loads(json_str)
    assert data["trace_id"] == "t1"
    assert len(data["spans"]) == 1
    assert data["spans"][0]["name"] == "Root"

def test_trace_duration():
    span = mod.Span("Test")
    span.start_time = 100.0
    span.end_time = 102.5
    assert span.duration == 2.5

def test_duration_ongoing():
    span = mod.Span("Test")
    span.start_time = 0.0
    # time.time() will be > 0
    assert span.duration > 0

def test_exporter_events_attributes():
    trace = mod.Trace("t2")
    span = mod.Span("Op")
    span.set_attribute("k", "v")
    span.add_event("evt")
    span.end()
    trace.add_span(span)
    data = json.loads(mod.SpanExporter.export_to_json(trace))
    assert data["spans"][0]["attributes"]["k"] == "v"
    assert len(data["spans"][0]["events"]) == 1
    assert data["spans"][0]["events"][0]["name"] == "evt"

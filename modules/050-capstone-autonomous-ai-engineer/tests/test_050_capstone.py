import os
import sys
import pytest
import importlib.util

# Load the implementation module dynamically
MODULE_PATH = os.path.join(os.path.dirname(__file__), "../04-implementation.py")
spec = importlib.util.spec_from_file_location("module_050_impl", MODULE_PATH)
impl = importlib.util.module_from_spec(spec)
sys.modules["module_050_impl"] = impl
spec.loader.exec_module(impl)

# --- Category 1: System Integration (4 tests) ---

def test_system_initialization():
    system = impl.AutonomousAIEngineerSystem()
    assert system.memory is not None
    assert system.tracer is not None
    assert system.sandbox is not None
    assert system.rag is not None

def test_memory_store_integration():
    system = impl.AutonomousAIEngineerSystem()
    system.memory.add("user", "Hello")
    assert len(system.memory.retrieve_recent(1)) == 1

def test_rag_engine_integration():
    system = impl.AutonomousAIEngineerSystem()
    system.rag.index_document("doc1", "FastAPI documentation")
    res = system.rag.query("FastAPI")
    assert "FastAPI documentation" in res

def test_sandbox_integration():
    system = impl.AutonomousAIEngineerSystem()
    system.sandbox.register_tool("echo", lambda x: x)
    assert system.sandbox.execute("echo", {"x": 42}) == 42

# --- Category 2: Autonomous Task Execution (4 tests) ---

def test_planner_creates_dag():
    system = impl.AutonomousAIEngineerSystem()
    plan = system.planner.plan("Build app")
    assert len(plan) > 0
    assert "dependencies" in plan[0]

def test_workflow_engine_execution():
    engine = impl.DAGWorkflowEngine()
    engine.add_node("A", "task A", [])
    engine.add_node("B", "task B", ["A"])
    
    executed = []
    engine.execute(lambda act: executed.append(act))
    assert executed == ["task A", "task B"]

def test_end_to_end_run():
    system = impl.AutonomousAIEngineerSystem()
    res = system.run("Build a web app")
    assert res["success"] is True
    assert res["steps_planned"] > 0

def test_memory_records_execution():
    system = impl.AutonomousAIEngineerSystem()
    system.run("Do something")
    memories = system.memory.retrieve_recent(10)
    system_messages = [m for m in memories if m["role"] == "system"]
    assert len(system_messages) > 0

# --- Category 3: Failover & Resilience (4 tests) ---

def test_sandbox_catches_exceptions():
    system = impl.AutonomousAIEngineerSystem()
    def crash(): raise ValueError("Boom")
    system.sandbox.register_tool("crash", crash)
    res = system.sandbox.execute("crash", {})
    assert "Tool execution failed" in res

def test_sandbox_missing_tool():
    system = impl.AutonomousAIEngineerSystem()
    with pytest.raises(ValueError):
        system.sandbox.execute("unknown", {})

def test_workflow_deadlock_detection():
    engine = impl.DAGWorkflowEngine()
    engine.add_node("A", "task A", ["B"])
    engine.add_node("B", "task B", ["A"])
    with pytest.raises(RuntimeError, match="DAG Deadlock"):
        engine.execute(lambda x: x)

def test_workflow_failure_propagation():
    engine = impl.DAGWorkflowEngine()
    engine.add_node("A", "task A", [])
    def fail_exec(action): raise Exception("Fail")
    res = engine.execute(fail_exec)
    assert res is False

# --- Category 4: Full System Observability (4 tests) ---

def test_tracer_records_spans():
    tracer = impl.OpenTelemetryTracer()
    span = tracer.start_span("test")
    tracer.end_span(span)
    traces = tracer.get_traces()
    assert len(traces) == 1
    assert traces[0]["status"] == "OK"

def test_tracer_nested_spans():
    tracer = impl.OpenTelemetryTracer()
    parent = tracer.start_span("parent")
    child = tracer.start_span("child", parent_id=parent)
    tracer.end_span(child)
    tracer.end_span(parent)
    traces = tracer.get_traces()
    assert len(traces) == 2
    assert traces[1]["parent_id"] == traces[0]["id"]

def test_message_bus_tracking():
    bus = impl.MultiAgentMessageBus()
    bus.subscribe("code", "agent_A")
    subscribers = bus.publish("code", "agent_B", "Here is code")
    assert "agent_A" in subscribers
    assert len(bus.messages) == 1

def test_full_system_telemetry_output():
    system = impl.AutonomousAIEngineerSystem()
    res = system.run("Test telemetry")
    traces = system.tracer.get_traces()
    # root, planning, exec, + individual actions
    assert res["traces"] == len(traces)
    assert len(traces) >= 3

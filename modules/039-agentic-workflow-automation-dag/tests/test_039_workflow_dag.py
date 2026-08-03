import pytest
import os
import sys
import importlib.util
from typing import Dict, Any

# Dynamic import of the implementation module
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '04-implementation.py'))
spec = importlib.util.spec_from_file_location("module_039_impl", file_path)
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)

# --- Category 1: Topological Sorting & Cycle Detection ---

def test_topological_sort_linear():
    graph = impl.StateGraph()
    graph.add_node(impl.WorkflowNode("A", lambda s: s))
    graph.add_node(impl.WorkflowNode("B", lambda s: s))
    graph.add_edge("A", "B")
    
    order = impl.TopologicalSorter.sort(graph)
    assert order == ["A", "B"]

def test_topological_sort_complex():
    graph = impl.StateGraph()
    for n in ["A", "B", "C", "D"]:
        graph.add_node(impl.WorkflowNode(n, lambda s: s))
    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_edge("B", "D")
    graph.add_edge("C", "D")
    
    order = impl.TopologicalSorter.sort(graph)
    assert order[0] == "A"
    assert order[-1] == "D"
    assert set(order[1:3]) == {"B", "C"}

def test_cycle_detection_true():
    graph = impl.StateGraph()
    graph.add_node(impl.WorkflowNode("A", lambda s: s))
    graph.add_node(impl.WorkflowNode("B", lambda s: s))
    graph.add_edge("A", "B")
    graph.add_edge("B", "A")
    
    assert impl.TopologicalSorter.has_cycle(graph) == True
    
def test_cycle_detection_false():
    graph = impl.StateGraph()
    graph.add_node(impl.WorkflowNode("A", lambda s: s))
    graph.add_node(impl.WorkflowNode("B", lambda s: s))
    graph.add_edge("A", "B")
    
    assert impl.TopologicalSorter.has_cycle(graph) == False

# --- Category 2: Graph Execution Engine ---

def test_execution_engine_success():
    graph = impl.StateGraph()
    def action_a(s): s['count'] = s.get('count', 0) + 1; return s
    def action_b(s): s['count'] *= 2; return s
    
    graph.add_node(impl.WorkflowNode("A", action_a))
    graph.add_node(impl.WorkflowNode("B", action_b))
    graph.add_edge("A", "B")
    
    engine = impl.DAGWorkflowEngine(graph)
    result = engine.execute({"count": 2})
    assert result['count'] == 6 # (2+1)*2

def test_execution_engine_retry_success():
    graph = impl.StateGraph()
    attempts = {'A': 0}
    def action_flaky(s):
        attempts['A'] += 1
        if attempts['A'] < 3:
            raise ValueError("Temporary failure")
        s['success'] = True
        return s
        
    graph.add_node(impl.WorkflowNode("A", action_flaky, max_retries=3))
    engine = impl.DAGWorkflowEngine(graph)
    result = engine.execute({})
    assert result['success'] == True
    assert attempts['A'] == 3

def test_execution_engine_retry_failure():
    graph = impl.StateGraph()
    def action_fail(s): raise ValueError("Permanent failure")
        
    graph.add_node(impl.WorkflowNode("A", action_fail, max_retries=2))
    engine = impl.DAGWorkflowEngine(graph)
    with pytest.raises(RuntimeError) as exc_info:
        engine.execute({})
    assert "failed after 2 retries" in str(exc_info.value)
    
def test_execution_engine_state_isolation():
    graph = impl.StateGraph()
    def action_a(s): s['list'].append(1); return s
    graph.add_node(impl.WorkflowNode("A", action_a))
    
    initial = {"list": []}
    engine = impl.DAGWorkflowEngine(graph)
    result = engine.execute(initial)
    assert result['list'] == [1]
    assert initial['list'] == [] # Should not mutate initial

# --- Category 3: Human-in-the-Loop Gates ---

def test_hitl_pause():
    graph = impl.StateGraph()
    graph.add_node(impl.WorkflowNode("A", lambda s: {**s, 'a': 1}, requires_human=True))
    
    engine = impl.DAGWorkflowEngine(graph)
    result = engine.execute({})
    assert result == {} # Paused before A executes
    assert engine.node_states["A"] == impl.NodeState.PAUSED

def test_hitl_resume():
    graph = impl.StateGraph()
    graph.add_node(impl.WorkflowNode("A", lambda s: {**s, 'a': 1}, requires_human=True))
    
    engine = impl.DAGWorkflowEngine(graph)
    engine.execute({})
    engine.hitl_gate.approve("A")
    result = engine.execute({}) # Resume
    assert result == {'a': 1}
    assert engine.node_states["A"] == impl.NodeState.COMPLETED

def test_hitl_multiple_gates():
    graph = impl.StateGraph()
    graph.add_node(impl.WorkflowNode("A", lambda s: {**s, 'a': 1}, requires_human=True))
    graph.add_node(impl.WorkflowNode("B", lambda s: {**s, 'b': 2}, requires_human=True))
    graph.add_edge("A", "B")
    
    engine = impl.DAGWorkflowEngine(graph)
    engine.execute({})
    engine.hitl_gate.approve("A")
    res1 = engine.execute({})
    assert res1 == {'a': 1}
    engine.hitl_gate.approve("B")
    res2 = engine.execute(res1)
    assert res2 == {'a': 1, 'b': 2}

def test_hitl_approval_idempotency():
    gate = impl.HumanInTheLoopGate()
    gate.request_approval("A")
    gate.approve("A")
    gate.approve("A") # Should not error
    assert gate.is_approved("A")

# --- Category 4: Checkpoint & Rollback ---

def test_checkpoint_save_restore():
    cm = impl.StateCheckpointManager()
    state = {"data": [1, 2, 3]}
    cm.save("A", state)
    state["data"].append(4)
    restored = cm.restore("A")
    assert restored["data"] == [1, 2, 3]

def test_checkpoint_isolation():
    cm = impl.StateCheckpointManager()
    cm.save("A", {"v": 1})
    restored1 = cm.restore("A")
    restored1["v"] = 2
    restored2 = cm.restore("A")
    assert restored2["v"] == 1

def test_engine_rollback_on_failure():
    graph = impl.StateGraph()
    def fail_action(s):
        s['mutated'] = True
        raise ValueError("Fail")
        
    graph.add_node(impl.WorkflowNode("A", fail_action, max_retries=0))
    engine = impl.DAGWorkflowEngine(graph)
    
    with pytest.raises(RuntimeError):
        engine.execute({"mutated": False})
        
    # The rollback happens internally before raising, but let's check checkpoint
    assert engine.checkpoint_manager.restore("A") == {"mutated": False}

def test_checkpoint_sequence():
    cm = impl.StateCheckpointManager()
    cm.save("A", {"step": 1})
    cm.save("B", {"step": 2})
    
    assert cm.restore("A") == {"step": 1}
    assert cm.restore("B") == {"step": 2}

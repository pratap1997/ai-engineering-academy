import pytest
import sys
import os
import importlib.util

# Use spec_from_file_location with a unique module name to avoid caching
# conflicts when running the full pytest suite across multiple modules.
_impl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '04-implementation.py'))
_spec = importlib.util.spec_from_file_location("module_027_impl", _impl_path)
module_027 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module_027)
sys.modules["module_027_impl"] = module_027

Message = module_027.Message
MessageBus = module_027.MessageBus
WorkerAgent = module_027.WorkerAgent
SupervisorAgent = module_027.SupervisorAgent
DebaterAgent = module_027.DebaterAgent
HierarchicalOrchestrator = module_027.HierarchicalOrchestrator
DebateOrchestrator = module_027.DebateOrchestrator
VotingMechanism = module_027.VotingMechanism


def mock_llm_fn(context: str) -> str:
    return "Mock response"

# ==============================================================================
# Category 1: Agent Communication (4)
# ==============================================================================

def test_message_bus_send_and_receive():
    bus = MessageBus()
    bus.register_agent("agent_A")
    bus.register_agent("agent_B")
    
    msg = Message("agent_A", "agent_B", "Hello", "task", 123)
    bus.send(msg)
    
    msgs = bus.receive("agent_B")
    assert len(msgs) == 1
    assert msgs[0].content == "Hello"
    assert len(bus.receive("agent_B")) == 0  # Inbox should be empty after reading

def test_message_bus_broadcast_reaches_all():
    bus = MessageBus()
    bus.register_agent("agent_A")
    bus.register_agent("agent_B")
    bus.register_agent("agent_C")
    
    msg = Message("agent_A", "*", "Broadcast msg", "broadcast", 123)
    bus.send(msg)
    
    assert len(bus.receive("agent_A")) == 0
    assert len(bus.receive("agent_B")) == 1
    msgs_c = bus.receive("agent_C")
    assert len(msgs_c) == 1
    assert msgs_c[0].content == "Broadcast msg"

def test_message_bus_agent_only_gets_own_messages():
    bus = MessageBus()
    bus.register_agent("agent_A")
    bus.register_agent("agent_B")
    
    msg = Message("agent_A", "agent_B", "Secret", "task", 123)
    bus.send(msg)
    
    assert len(bus.receive("agent_A")) == 0

def test_message_type_categorization():
    bus = MessageBus()
    bus.register_agent("A")
    bus.register_agent("B")
    
    msg1 = Message("A", "B", "Do this", "task", 1)
    msg2 = Message("A", "B", "Stop", "terminate", 2)
    bus.send(msg1)
    bus.send(msg2)
    
    msgs = bus.receive("B")
    assert [m.message_type for m in msgs] == ["task", "terminate"]

# ==============================================================================
# Category 2: Orchestration Patterns (4)
# ==============================================================================

def test_hierarchical_orchestrator_assigns_tasks():
    bus = MessageBus()
    w1 = WorkerAgent("w1", "coder", mock_llm_fn, bus)
    sup = SupervisorAgent("sup", ["w1"], mock_llm_fn, bus)
    orch = HierarchicalOrchestrator(sup, [w1], bus)
    
    # Check assignment
    sup.assign_task("Write code", "w1")
    msgs = bus.receive("w1")
    assert len(msgs) == 1
    assert msgs[0].content == "Write code"

def test_supervisor_synthesizes_worker_results():
    bus = MessageBus()
    w1 = WorkerAgent("w1", "coder", mock_llm_fn, bus)
    sup = SupervisorAgent("sup", ["w1"], mock_llm_fn, bus)
    
    # Mocking result
    bus.send(Message("w1", "sup", "Code result", "result", 1))
    sup.collect_results()
    assert "w1" in sup.results
    
    synth = sup.synthesize(sup.results)
    assert synth == "Mock response"

def test_debate_orchestrator_runs_multiple_rounds():
    bus = MessageBus()
    d1 = DebaterAgent("d1", "pro", mock_llm_fn, bus)
    d2 = DebaterAgent("d2", "con", mock_llm_fn, bus)
    
    orch = DebateOrchestrator([d1, d2], rounds=2, bus=bus)
    result = orch.run_debate("Is AI good?")
    
    # 2 agents * 2 rounds = 4 messages in transcript
    assert len(result["transcript"]) == 4

def test_debate_declares_winner():
    bus = MessageBus()
    d1 = DebaterAgent("d1", "pro", mock_llm_fn, bus)
    d2 = DebaterAgent("d2", "con", mock_llm_fn, bus)
    
    orch = DebateOrchestrator([d1, d2], rounds=1, bus=bus)
    result = orch.run_debate("Topic")
    assert "winner" in result
    assert result["winner"] == "Mock response"

# ==============================================================================
# Category 3: Task Assignment (4)
# ==============================================================================

def test_worker_receives_and_processes_task():
    bus = MessageBus()
    bus.register_agent("sup")
    w1 = WorkerAgent("w1", "coder", mock_llm_fn, bus)
    
    bus.send(Message("sup", "w1", "Code this", "task", 1))
    w1.run_step()
    
    msgs_to_sup = bus.receive("sup")
    assert len(msgs_to_sup) == 1
    assert msgs_to_sup[0].message_type == "result"
    assert msgs_to_sup[0].content == "Mock response"

def test_multi_worker_parallel_task_handling():
    bus = MessageBus()
    w1 = WorkerAgent("w1", "coder", mock_llm_fn, bus)
    w2 = WorkerAgent("w2", "reviewer", mock_llm_fn, bus)
    sup = SupervisorAgent("sup", ["w1", "w2"], mock_llm_fn, bus)
    
    orch = HierarchicalOrchestrator(sup, [w1, w2], bus)
    res = orch.run("Build app")
    
    # Both should have sent results back to supervisor during run
    assert "w1" in sup.results
    assert "w2" in sup.results

def test_orchestrator_handles_worker_timeout():
    # Simulate an orchestrator reaching max steps (timeout)
    bus = MessageBus()
    w1 = WorkerAgent("w1", "coder", mock_llm_fn, bus)
    
    # Never sending result to supervisor so it loops
    class LazyWorker(WorkerAgent):
        def run_step(self): return True
        
    lazy_w1 = LazyWorker("w1", "coder", mock_llm_fn, bus)
    sup = SupervisorAgent("sup", ["w1"], mock_llm_fn, bus)
    
    orch = HierarchicalOrchestrator(sup, [lazy_w1], bus)
    res = orch.run("Task")
    
    # The while active and steps < timeout should terminate it
    assert res["steps"] == 20

def test_result_collection_timeout():
    bus = MessageBus()
    sup = SupervisorAgent("sup", ["w1"], mock_llm_fn, bus)
    sup.tasks_assigned = 1
    
    # No result sent
    active = sup.run_step()
    assert active is True # Still active, waiting for results
    assert len(sup.results) == 0

# ==============================================================================
# Category 4: Coordination Math (4)
# ==============================================================================

def test_voting_simple_majority():
    v = VotingMechanism(["A", "B", "C"])
    prefs = {
        "A": ["opt1", "opt2"],
        "B": ["opt1", "opt3"],
        "C": ["opt2", "opt1"]
    }
    winner = v.collect_votes(["opt1", "opt2", "opt3"], prefs)
    assert winner == "opt1"

def test_condorcet_winner_detected():
    v = VotingMechanism(["A", "B", "C"])
    # opt1 beats opt2 and opt3 in head-to-head
    prefs = {
        "A": ["opt1", "opt2", "opt3"],
        "B": ["opt2", "opt1", "opt3"],
        "C": ["opt1", "opt3", "opt2"]
    }
    winner = v.condorcet_winner(prefs)
    assert winner == "opt1"

def test_condorcet_no_winner_returns_none():
    v = VotingMechanism(["A", "B", "C"])
    # Condorcet paradox: 1 beats 2, 2 beats 3, 3 beats 1
    prefs = {
        "A": ["opt1", "opt2", "opt3"],
        "B": ["opt2", "opt3", "opt1"],
        "C": ["opt3", "opt1", "opt2"]
    }
    winner = v.condorcet_winner(prefs)
    assert winner is None

def test_message_count_scales_correctly():
    bus = MessageBus()
    for i in range(5):
        bus.register_agent(f"a{i}")
        
    bus.broadcast("a0", "hello")
    
    # 4 other agents should receive it
    received = sum(len(bus.receive(f"a{i}")) for i in range(1, 5))
    assert received == 4

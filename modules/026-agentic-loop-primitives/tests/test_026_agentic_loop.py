import pytest
import os
import sys
import importlib.util

# Use spec_from_file_location with unique name to avoid caching conflicts
# when running the full pytest suite across multiple modules.
_impl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '04-implementation.py'))
_spec = importlib.util.spec_from_file_location("module_026_impl", _impl_path)
impl = importlib.util.module_from_spec(_spec)
sys.modules["module_026_impl"] = impl
_spec.loader.exec_module(impl)

Tool = impl.Tool
ToolRegistry = impl.ToolRegistry
Thought = impl.Thought
Action = impl.Action
Observation = impl.Observation
TrajectoryStep = impl.TrajectoryStep
ReActAgent = impl.ReActAgent
UCBToolSelector = impl.UCBToolSelector

# --- Category 1: Tool System ---

def test_tool_execution_returns_string():
    tool = Tool("Test", "desc", lambda x: x*2, {})
    res = tool.execute(x=5)
    assert isinstance(res, str)
    assert res == "10"
    
def test_tool_registry_register_and_get():
    reg = ToolRegistry()
    t = Tool("T1", "desc", lambda: 1, {})
    reg.register(t)
    fetched = reg.get("T1")
    assert fetched.name == "T1"
    
def test_tool_registry_list_tools_format():
    reg = ToolRegistry()
    reg.register(Tool("T1", "desc1", lambda: 1, {}))
    reg.register(Tool("T2", "desc2", lambda: 2, {}))
    out = reg.list_tools()
    assert "- T1: desc1" in out
    assert "- T2: desc2" in out
    
def test_tool_registry_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="not found"):
        reg.get("Unknown")

# --- Category 2: ReAct Components ---

def test_action_parse_valid_format():
    text = 'Action: Calc\nAction Input: {"x": 5}'
    action = Action.parse(text)
    assert action.tool_name == "Calc"
    assert action.args == {"x": 5}
    
def test_action_parse_with_args():
    text = 'Action: Func({"val": "test"})'
    action = Action.parse(text)
    assert action.tool_name == "Func"
    assert action.args == {"val": "test"}
    
def test_trajectory_step_stores_components():
    t = Thought("thinking")
    a = Action("tool", {})
    o = Observation("result", True)
    step = TrajectoryStep(t, a, o)
    assert step.thought.content == "thinking"
    assert step.action.tool_name == "tool"
    assert step.observation.success is True
    
def test_react_agent_builds_correct_prompt():
    reg = ToolRegistry()
    agent = ReActAgent(lambda x: "", [])
    prompt = agent._build_prompt("Solve this", [])
    assert "Task: Solve this" in prompt
    assert "Thought:" in prompt

# --- Category 3: Agent Behavior ---

def test_agent_stops_at_finish_action():
    def mock_llm(prompt):
        return 'Thought: done\nAction: FINISH\nAction Input: {"answer": "42"}'
    agent = ReActAgent(mock_llm, [])
    res = agent.run("task")
    assert res["success"] is True
    assert res["answer"] == "42"
    assert res["steps"] == 1
    
def test_agent_stops_at_max_steps():
    def mock_llm(prompt):
        return 'Thought: stuck\nAction: Wait\nAction Input: {}'
    agent = ReActAgent(mock_llm, [], max_steps=3)
    res = agent.run("task")
    assert res["success"] is False
    assert res["steps"] == 3
    
def test_agent_uses_tool_output_in_next_step():
    def mock_llm(prompt):
        if "Observation: 100" in prompt:
            return 'Thought: got it\nAction: FINISH\nAction Input: {"answer": "100"}'
        return 'Thought: hmm\nAction: T1\nAction Input: {}'
    
    t1 = Tool("T1", "desc", lambda: "100", {})
    agent = ReActAgent(mock_llm, [t1])
    res = agent.run("task")
    assert res["success"] is True
    assert res["steps"] == 2
    
def test_agent_returns_trajectory():
    def mock_llm(prompt):
        return 'Thought: a\nAction: FINISH\nAction Input: {"answer": "b"}'
    agent = ReActAgent(mock_llm, [])
    res = agent.run("t")
    assert len(res["trajectory"]) == 1
    assert res["trajectory"][0].action.tool_name == "FINISH"

# --- Category 4: UCB Tool Selector ---

def test_ucb_initial_selects_unplayed_tool():
    ucb = UCBToolSelector(["A", "B"])
    t1 = ucb.select()
    assert t1 in ["A", "B"]
    ucb.update(t1, 1.0)
    t2 = ucb.select()
    assert t2 != t1
    
def test_ucb_update_increases_play_count():
    ucb = UCBToolSelector(["A"])
    ucb.select()
    ucb.update("A", 1.0)
    assert ucb.counts["A"] == 1
    
def test_ucb_score_decreases_after_selection():
    ucb = UCBToolSelector(["A", "B"])
    ucb.update("A", 1.0) 
    ucb.update("B", 1.0) 
    score1 = ucb.ucb_scores()["A"]
    ucb.update("A", 1.0) 
    score2 = ucb.ucb_scores()["A"]
    assert score2 < score1
    
def test_ucb_converges_to_best_tool():
    ucb = UCBToolSelector(["A", "B"], c=0.5)
    for _ in range(50):
        t = ucb.select()
        reward = 1.0 if t == "A" else 0.0
        ucb.update(t, reward)
    
    assert ucb.counts["A"] > ucb.counts["B"]

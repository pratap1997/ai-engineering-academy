import pytest
import math
import sys
from pathlib import Path
import importlib.util

module_path = Path(__file__).parent.parent / "04-implementation.py"
spec = importlib.util.spec_from_file_location("module_046_impl", module_path)
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)

TreeNode = impl.TreeNode
MCTSSearchEngine = impl.MCTSSearchEngine
RolloutSimulator = impl.RolloutSimulator
TreeOfThoughtsPlanner = impl.TreeOfThoughtsPlanner
ThoughtEvaluator = impl.ThoughtEvaluator

# -- UCT Selection Math (4 tests) --
def test_uct_math_explore():
    engine = MCTSSearchEngine(exploration_constant=1.0)
    root = TreeNode(state=0)
    root.visits = 10
    child1 = TreeNode(state=1, parent=root)
    child1.visits = 1
    child1.value = 1.0
    root.children = [child1]
    best = engine.best_child(root)
    assert best == child1

def test_uct_math_exploit():
    engine = MCTSSearchEngine(exploration_constant=0.0) # pure exploit
    root = TreeNode(state=0)
    root.visits = 10
    c1 = TreeNode(state=1, parent=root)
    c1.visits = 5; c1.value = 10.0 # avg 2.0
    c2 = TreeNode(state=2, parent=root)
    c2.visits = 5; c2.value = 5.0  # avg 1.0
    root.children = [c1, c2]
    assert engine.best_child(root) == c1

def test_uct_zero_visits():
    engine = MCTSSearchEngine()
    root = TreeNode(state=0)
    c1 = TreeNode(state=1, parent=root)
    c1.visits = 0
    root.children = [c1]
    assert engine.best_child(root) == c1

def test_uct_math_balance():
    engine = MCTSSearchEngine(exploration_constant=1.0)
    root = TreeNode(state=0)
    root.visits = 100
    c1 = TreeNode(state=1, parent=root)
    c1.visits = 50; c1.value = 25.0
    c2 = TreeNode(state=2, parent=root)
    c2.visits = 2; c2.value = 0.5
    root.children = [c1, c2]
    # score1 = 0.5 + sqrt(ln100 / 50) = 0.5 + sqrt(4.6/50) = 0.5 + 0.3 = 0.8
    # score2 = 0.25 + sqrt(ln100 / 2) = 0.25 + sqrt(4.6/2) = 0.25 + 1.51 = 1.76
    assert engine.best_child(root) == c2

# -- Tree Expansion & Traversal (4 tests) --
def test_tree_node_init():
    node = TreeNode(state=5)
    assert node.state == 5
    assert node.parent is None
    assert node.is_terminal is False

def test_is_fully_expanded():
    node = TreeNode(state=0)
    assert node.is_fully_expanded(0) is True
    assert node.is_fully_expanded(1) is False
    node.children.append(TreeNode(state=1))
    assert node.is_fully_expanded(1) is True

def test_engine_expand():
    def get_actions(s): return ['a', 'b']
    def apply_action(s, a): return s + a
    engine = MCTSSearchEngine(get_possible_actions=get_actions, apply_action=apply_action, is_terminal=lambda s: False)
    root = TreeNode(state="root_")
    child = engine.expand(root)
    assert child.state in ["root_a", "root_b"]
    assert child.parent == root

def test_engine_select():
    def get_actions(s): return ['a']
    def apply_action(s, a): return s + a
    engine = MCTSSearchEngine(get_possible_actions=get_actions, apply_action=apply_action, is_terminal=lambda s: False)
    root = TreeNode(state="start")
    leaf = engine.select(root)
    assert leaf.state == "starta"

# -- Value Backpropagation (4 tests) --
def test_backpropagate_single():
    engine = MCTSSearchEngine()
    node = TreeNode(state=0)
    engine.backpropagate(node, 1.0)
    assert node.visits == 1
    assert node.value == 1.0

def test_backpropagate_chain():
    engine = MCTSSearchEngine()
    root = TreeNode(state=0)
    child = TreeNode(state=1, parent=root)
    grandchild = TreeNode(state=2, parent=child)
    engine.backpropagate(grandchild, 5.0)
    assert root.visits == 1
    assert root.value == 5.0
    assert child.visits == 1
    assert grandchild.value == 5.0

def test_backpropagate_accumulate():
    engine = MCTSSearchEngine()
    node = TreeNode(state=0)
    engine.backpropagate(node, 1.0)
    engine.backpropagate(node, 0.5)
    assert node.visits == 2
    assert node.value == 1.5

def test_thought_evaluator():
    ev = ThoughtEvaluator()
    assert ev.evaluate([1, 2]) == 1.0

# -- MCTS Planner Execution (4 tests) --
def get_actions(s): return [1, 2] if s < 5 else []
def apply_action(s, a): return s + a
def evaluate(s): return float(s)
def is_terminal(s): return s >= 5

def test_simulator():
    sim = RolloutSimulator(max_depth=3)
    node = TreeNode(state=0)
    res = sim.simulate(node, get_actions, apply_action, evaluate)
    assert res > 0.0

def test_planner_init():
    engine = MCTSSearchEngine(get_possible_actions=get_actions, apply_action=apply_action, evaluate=evaluate, is_terminal=is_terminal)
    sim = RolloutSimulator()
    planner = TreeOfThoughtsPlanner(engine, sim)
    assert planner.engine == engine

def test_planner_execution():
    engine = MCTSSearchEngine(get_possible_actions=get_actions, apply_action=apply_action, evaluate=evaluate, is_terminal=is_terminal)
    sim = RolloutSimulator()
    planner = TreeOfThoughtsPlanner(engine, sim)
    plan = planner.plan(initial_state=0, num_iterations=10)
    assert len(plan) > 0

def test_terminal_node():
    engine = MCTSSearchEngine(get_possible_actions=get_actions, apply_action=apply_action, evaluate=evaluate, is_terminal=is_terminal)
    root = TreeNode(state=5)
    root.is_terminal = True
    leaf = engine.select(root)
    assert leaf == root

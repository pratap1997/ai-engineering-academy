import math
import random
from typing import List, Optional, Callable, Any

class TreeNode:
    def __init__(self, state: Any, parent: Optional['TreeNode'] = None, action: Any = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List['TreeNode'] = []
        self.visits = 0
        self.value = 0.0
        self.is_terminal = False

    def is_fully_expanded(self, possible_actions: int) -> bool:
        return len(self.children) == possible_actions

class ThoughtEvaluator:
    def evaluate(self, state: Any) -> float:
        if isinstance(state, list):
            return sum(state) / (len(state) + 1)
        return random.random()

class RolloutSimulator:
    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        
    def simulate(self, node: TreeNode, get_possible_actions: Callable, apply_action: Callable, evaluate: Callable) -> float:
        current_state = node.state
        depth = 0
        while depth < self.max_depth:
            actions = get_possible_actions(current_state)
            if not actions:
                break
            action = random.choice(actions)
            current_state = apply_action(current_state, action)
            depth += 1
        return evaluate(current_state)

class MCTSSearchEngine:
    def __init__(self, exploration_constant: float = 1.414, get_possible_actions=None, apply_action=None, evaluate=None, is_terminal=None):
        self.exploration_constant = exploration_constant
        self.get_possible_actions = get_possible_actions
        self.apply_action = apply_action
        self.evaluate = evaluate
        self.is_terminal = is_terminal
        
    def select(self, node: TreeNode) -> TreeNode:
        while not node.is_terminal and len(self.get_possible_actions(node.state)) > 0:
            if not node.is_fully_expanded(len(self.get_possible_actions(node.state))):
                return self.expand(node)
            else:
                node = self.best_child(node)
        return node
        
    def expand(self, node: TreeNode) -> TreeNode:
        actions = self.get_possible_actions(node.state)
        tried_actions = [child.action for child in node.children]
        untried_actions = [a for a in actions if a not in tried_actions]
        
        if not untried_actions:
            return node
            
        action = random.choice(untried_actions)
        next_state = self.apply_action(node.state, action)
        child = TreeNode(state=next_state, parent=node, action=action)
        child.is_terminal = self.is_terminal(next_state)
        node.children.append(child)
        return child
        
    def backpropagate(self, node: TreeNode, reward: float):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
            
    def best_child(self, node: TreeNode) -> TreeNode:
        best_score = -float('inf')
        best_children = []
        for child in node.children:
            if child.visits == 0:
                score = float('inf')
            else:
                exploit = child.value / child.visits
                explore = self.exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
                score = exploit + explore
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
        if not best_children:
            return node
        return random.choice(best_children)

class TreeOfThoughtsPlanner:
    def __init__(self, engine: MCTSSearchEngine, simulator: RolloutSimulator):
        self.engine = engine
        self.simulator = simulator
        
    def plan(self, initial_state: Any, num_iterations: int = 100) -> List[Any]:
        root = TreeNode(state=initial_state)
        root.is_terminal = self.engine.is_terminal(initial_state)
        
        for _ in range(num_iterations):
            leaf = self.engine.select(root)
            reward = self.simulator.simulate(leaf, self.engine.get_possible_actions, self.engine.apply_action, self.engine.evaluate)
            self.engine.backpropagate(leaf, reward)
            
        plan = []
        curr = root
        while curr.children:
            best = max(curr.children, key=lambda c: c.visits if c.visits > 0 else -1)
            plan.append(best.action)
            curr = best
            
        return plan

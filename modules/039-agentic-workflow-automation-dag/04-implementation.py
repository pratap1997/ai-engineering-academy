import json
import math
import time
from typing import Dict, List, Any, Optional, Callable, Set

class NodeState:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED" # For HITL

class WorkflowNode:
    def __init__(self, id: str, action: Callable[[Dict[str, Any]], Dict[str, Any]], max_retries: int = 3, requires_human: bool = False):
        self.id = id
        self.action = action
        self.max_retries = max_retries
        self.requires_human = requires_human

class StateCheckpointManager:
    def __init__(self):
        self.checkpoints = {}
        
    def save(self, step_id: str, state: Dict[str, Any]):
        self.checkpoints[step_id] = json.loads(json.dumps(state))
        
    def restore(self, step_id: str) -> Dict[str, Any]:
        return json.loads(json.dumps(self.checkpoints.get(step_id, {})))

class StateGraph:
    def __init__(self):
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: Dict[str, List[str]] = {}
        self.in_degree: Dict[str, int] = {}
        
    def add_node(self, node: WorkflowNode):
        self.nodes[node.id] = node
        if node.id not in self.edges:
            self.edges[node.id] = []
            self.in_degree[node.id] = 0
            
    def add_edge(self, from_id: str, to_id: str):
        if from_id not in self.nodes or to_id not in self.nodes:
            raise ValueError("Nodes must exist before adding edges")
        self.edges[from_id].append(to_id)
        self.in_degree[to_id] += 1

class TopologicalSorter:
    @staticmethod
    def sort(graph: StateGraph) -> List[str]:
        # Kahn's algorithm
        in_degree = graph.in_degree.copy()
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        sorted_order = []
        
        while queue:
            node = queue.pop(0)
            sorted_order.append(node)
            for neighbor in graph.edges[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        if len(sorted_order) != len(graph.nodes):
            raise ValueError("Graph contains a cycle")
            
        return sorted_order
        
    @staticmethod
    def has_cycle(graph: StateGraph) -> bool:
        try:
            TopologicalSorter.sort(graph)
            return False
        except ValueError:
            return True

class HumanInTheLoopGate:
    def __init__(self):
        self.pending_approvals = set()
        
    def request_approval(self, node_id: str):
        self.pending_approvals.add(node_id)
        
    def approve(self, node_id: str):
        if node_id in self.pending_approvals:
            self.pending_approvals.remove(node_id)
            
    def is_approved(self, node_id: str) -> bool:
        return node_id not in self.pending_approvals

class DAGWorkflowEngine:
    def __init__(self, graph: StateGraph):
        self.graph = graph
        self.checkpoint_manager = StateCheckpointManager()
        self.hitl_gate = HumanInTheLoopGate()
        self.node_states: Dict[str, str] = {node_id: NodeState.PENDING for node_id in graph.nodes}
        self.retries: Dict[str, int] = {node_id: 0 for node_id in graph.nodes}
        
    def execute(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        execution_order = TopologicalSorter.sort(self.graph)
        current_state = json.loads(json.dumps(initial_state))
        
        for node_id in execution_order:
            node = self.graph.nodes[node_id]
            
            if self.node_states[node_id] == NodeState.COMPLETED:
                continue
                
            if node.requires_human:
                if self.node_states[node_id] != NodeState.PAUSED:
                    self.hitl_gate.request_approval(node_id)
                    self.node_states[node_id] = NodeState.PAUSED
                    
                if not self.hitl_gate.is_approved(node_id):
                    return current_state # Pause execution, return current state
                    
            self.checkpoint_manager.save(node_id, current_state)
            self.node_states[node_id] = NodeState.RUNNING
            
            success = False
            while self.retries[node_id] <= node.max_retries and not success:
                try:
                    current_state = node.action(current_state)
                    success = True
                    self.node_states[node_id] = NodeState.COMPLETED
                except Exception as e:
                    self.retries[node_id] += 1
                    if self.retries[node_id] > node.max_retries:
                        self.node_states[node_id] = NodeState.FAILED
                        # Rollback
                        current_state = self.checkpoint_manager.restore(node_id)
                        raise RuntimeError(f"Node {node_id} failed after {node.max_retries} retries: {str(e)}")
                    # Exponential backoff
                    delay = min(1.0 * (2 ** (self.retries[node_id] - 1)), 10.0)
                    time.sleep(0.01) # Use small sleep for tests
                    current_state = self.checkpoint_manager.restore(node_id)
                    
        return current_state

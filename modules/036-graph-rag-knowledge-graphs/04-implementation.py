import math
import collections
from typing import Dict, List, Set, Tuple, Optional, Any

class Entity:
    def __init__(self, id: str, type: str, attributes: Dict[str, str] = None):
        self.id = id
        self.type = type
        self.attributes = attributes or {}

class Relation:
    def __init__(self, source: str, target: str, type: str, weight: float = 1.0):
        self.source = source
        self.target = target
        self.type = type
        self.weight = weight

class KnowledgeGraph:
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.adjacency: Dict[str, Dict[str, float]] = collections.defaultdict(dict)

    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity

    def add_relation(self, relation: Relation):
        if relation.source not in self.entities or relation.target not in self.entities:
            raise ValueError("Entities must exist before adding relation")
        self.relations.append(relation)
        self.adjacency[relation.source][relation.target] = relation.weight
        
    def get_neighbors(self, entity_id: str) -> Dict[str, float]:
        return self.adjacency.get(entity_id, {})
        
    def get_nodes(self) -> List[str]:
        return list(self.entities.keys())

class GraphExtractor:
    def __init__(self):
        self.mock_responses = {}
        
    def add_mock_response(self, text: str, entities: List[Entity], relations: List[Relation]):
        self.mock_responses[text] = (entities, relations)
        
    def extract(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        return self.mock_responses.get(text, ([], []))

class PageRankRanker:
    def __init__(self, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    def rank(self, graph: KnowledgeGraph) -> Dict[str, float]:
        nodes = graph.get_nodes()
        if not nodes:
            return {}
            
        N = len(nodes)
        ranks = {node: 1.0 / N for node in nodes}
        out_degree = {node: sum(graph.get_neighbors(node).values()) for node in nodes}
        
        for _ in range(self.max_iter):
            new_ranks = {node: (1.0 - self.damping) / N for node in nodes}
            diff = 0.0
            
            for u in nodes:
                neighbors = graph.get_neighbors(u)
                for v, weight in neighbors.items():
                    if out_degree[u] > 0:
                        new_ranks[v] += self.damping * ranks[u] * (weight / out_degree[u])
                    
            dangling_sum = sum(ranks[u] for u in nodes if out_degree[u] == 0)
            if dangling_sum > 0:
                for v in nodes:
                    new_ranks[v] += self.damping * dangling_sum / N
                    
            for node in nodes:
                diff += abs(new_ranks[node] - ranks[node])
                
            ranks = new_ranks
            if diff < self.tol:
                break
                
        return ranks

class LeidenCommunityDetector:
    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution
        
    def detect(self, graph: KnowledgeGraph) -> Dict[str, int]:
        nodes = graph.get_nodes()
        communities = {node: i for i, node in enumerate(nodes)}
        
        if not nodes:
            return communities
            
        changed = True
        while changed:
            changed = False
            for u in nodes:
                neighbors = graph.get_neighbors(u)
                if not neighbors:
                    continue
                neighbor_comms = [communities[v] for v in neighbors.keys()]
                if not neighbor_comms:
                    continue
                best_comm = max(set(neighbor_comms), key=neighbor_comms.count)
                if communities[u] != best_comm:
                    communities[u] = best_comm
                    changed = True
                    
        unique_comms = list(set(communities.values()))
        return {node: unique_comms.index(c) for node, c in communities.items()}

class GraphRAGRetriever:
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.ranker = PageRankRanker()
        
    def global_search(self) -> List[Tuple[str, float]]:
        ranks = self.ranker.rank(self.graph)
        return sorted(ranks.items(), key=lambda x: x[1], reverse=True)
        
    def local_search(self, start_entity: str, hops: int = 2) -> Dict[str, float]:
        visited = {start_entity: 1.0}
        frontier = {start_entity: 1.0}
        
        for _ in range(hops):
            next_frontier = {}
            for u, score in frontier.items():
                neighbors = self.graph.get_neighbors(u)
                for v, weight in neighbors.items():
                    if v not in visited:
                        new_score = score * weight
                        next_frontier[v] = next_frontier.get(v, 0.0) + new_score
                        visited[v] = visited.get(v, 0.0) + new_score
            frontier = next_frontier
            
        return visited

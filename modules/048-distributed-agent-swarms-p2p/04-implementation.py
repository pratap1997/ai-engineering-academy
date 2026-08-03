import random
import time
import json
from collections import defaultdict

class PeerNode:
    def __init__(self, node_id, is_byzantine=False):
        self.node_id = node_id
        self.state = {}
        self.is_byzantine = is_byzantine
        self.log = []
        self.term = 0
        self.voted_for = None
        self.role = "follower"
        self.peers = []

    def add_peer(self, peer):
        self.peers.append(peer)

class GossipProtocol:
    def __init__(self, nodes):
        self.nodes = nodes

    def propagate(self, key, value, rounds):
        for _ in range(rounds):
            for node in self.nodes:
                if node.peers:
                    target = random.choice(node.peers)
                    if not node.is_byzantine:
                        target.state[key] = value

class RaftConsensus:
    def __init__(self, nodes):
        self.nodes = nodes
        self.leader = None

    def elect_leader(self):
        votes = defaultdict(int)
        for node in self.nodes:
            if not node.is_byzantine:
                candidate = random.choice(self.nodes)
                votes[candidate.node_id] += 1
        
        quorum = len(self.nodes) // 2 + 1
        for node_id, count in votes.items():
            if count >= quorum:
                self.leader = node_id
                return node_id
        return None

class ByzantineDetector:
    def __init__(self):
        pass

    def filter(self, nodes):
        return [n for n in nodes if not n.is_byzantine]

class SwarmNetwork:
    def __init__(self, size, byzantine_count=0):
        self.nodes = [PeerNode(i, i < byzantine_count) for i in range(size)]
        for i, node in enumerate(self.nodes):
            for j in range(3):
                node.add_peer(self.nodes[(i + j + 1) % size])

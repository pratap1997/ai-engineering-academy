import time
import math
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Memory:
    """A single memory unit with content, metadata, and importance."""
    id: str
    content: str
    embedding: list[float]  # simple vector
    timestamp: float
    access_count: int = 0
    importance: float = 1.0
    memory_type: str = "episodic"  # episodic, semantic, procedural
    tags: list[str] = field(default_factory=list)

class WorkingMemory:
    """Fixed-capacity context window (FIFO queue)."""
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.queue = []

    def add(self, content: str) -> None:
        if self.is_full():
            self.queue.pop(0)
        self.queue.append(content)

    def get_all(self) -> list[str]:
        return list(self.queue)

    def clear(self) -> None:
        self.queue.clear()

    def is_full(self) -> bool:
        return len(self.queue) >= self.capacity

class EmbeddingModel:
    """Simple bag-of-words embedding (no external dependencies)."""
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        
    def embed(self, text: str) -> list[float]:
        words = text.lower().replace('.', '').replace(',', '').split()
        vector = [0.0] * self.vocab_size
        for w in words:
            idx = sum(ord(c) for c in w) % self.vocab_size
            vector[idx] += 1.0
        
        # Normalize
        norm = math.sqrt(sum(v*v for v in vector))
        if norm > 0:
            return [v/norm for v in vector]
        return vector

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x*y for x,y in zip(a,b))
        norm_a = math.sqrt(sum(x*x for x in a))
        norm_b = math.sqrt(sum(y*y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

class MemoryStore:
    """Long-term memory store with vector similarity retrieval."""
    def __init__(self, embedder: EmbeddingModel, max_memories: int = 1000):
        self.embedder = embedder
        self.max_memories = max_memories
        self.memories = {}
        self._next_id = 1

    def store(self, content: str, memory_type: str = "episodic", importance: float = 1.0, tags: list = None) -> Memory:
        mem_id = str(self._next_id)
        self._next_id += 1
        embedding = self.embedder.embed(content)
        if tags is None:
            tags = []
        mem = Memory(id=mem_id, content=content, embedding=embedding, 
                     timestamp=time.time(), importance=importance, 
                     memory_type=memory_type, tags=tags)
        self.memories[mem_id] = mem
        return mem

    def retrieve(self, query: str, k: int = 5, memory_type: str = None) -> list[Memory]:
        q_emb = self.embedder.embed(query)
        scored = []
        for mem in self.memories.values():
            if memory_type and mem.memory_type != memory_type:
                continue
            sim = self.embedder.cosine_similarity(q_emb, mem.embedding)
            score = sim * mem.importance
            scored.append((score, mem))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [x[1] for x in scored[:k]]
        for r in results:
            r.access_count += 1
        return results

    def update(self, memory_id: str, new_content: str) -> Memory:
        if memory_id in self.memories:
            self.memories[memory_id].content = new_content
            self.memories[memory_id].embedding = self.embedder.embed(new_content)
            self.memories[memory_id].timestamp = time.time()
            return self.memories[memory_id]
        return None

    def delete(self, memory_id: str) -> bool:
        if memory_id in self.memories:
            del self.memories[memory_id]
            return True
        return False

    def forget_curve(self, elapsed_hours: float, strength: float = 1.0) -> float:
        # R = e^{-t/S}
        return math.exp(-elapsed_hours / max(0.1, strength))

    def consolidate(self) -> int:
        current_time = time.time()
        to_remove = []
        for mem_id, mem in self.memories.items():
            elapsed_hours = (current_time - mem.timestamp) / 3600.0
            strength = mem.importance + (mem.access_count * 0.1)
            retention = self.forget_curve(elapsed_hours, strength)
            if retention < 0.1: # Threshold to forget
                to_remove.append(mem_id)
                
        for mem_id in to_remove:
            self.delete(mem_id)
        return len(to_remove)

    def stats(self) -> dict:
        return {
            "total_memories": len(self.memories),
            "by_type": {
                "episodic": len([m for m in self.memories.values() if m.memory_type == "episodic"]),
                "semantic": len([m for m in self.memories.values() if m.memory_type == "semantic"])
            }
        }

class AgentMemorySystem:
    """Complete agent memory system combining working + long-term memory."""
    def __init__(self, working_capacity: int = 10, long_term_capacity: int = 500):
        self.working_memory = WorkingMemory(capacity=working_capacity)
        self.embedder = EmbeddingModel()
        self.long_term_memory = MemoryStore(embedder=self.embedder, max_memories=long_term_capacity)

    def perceive(self, observation: str) -> None:
        self.working_memory.add(observation)

    def remember(self, content: str, importance: float = 1.0) -> Memory:
        return self.long_term_memory.store(content, importance=importance)

    def recall(self, query: str, k: int = 3) -> list[Memory]:
        return self.long_term_memory.retrieve(query, k=k)

    def build_context(self, query: str) -> str:
        relevant_memories = self.recall(query, k=2)
        context_str = "Recent Context:\n"
        context_str += "\n".join(self.working_memory.get_all())
        context_str += "\n\nRelevant Past Memories:\n"
        context_str += "\n".join([m.content for m in relevant_memories])
        return context_str

    def reflect(self) -> list[str]:
        return ["Need more data to form insights."]

    def stats(self) -> dict:
        return {
            "working_memory_items": len(self.working_memory.get_all()),
            "long_term_memory_items": self.long_term_memory.stats()["total_memories"]
        }

class HopfieldNetwork:
    """Hopfield associative memory network (from-scratch)."""
    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        self.weights = [[0.0] * n_neurons for _ in range(n_neurons)]

    def store_pattern(self, pattern: list[int]) -> None:
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i != j:
                    self.weights[i][j] += pattern[i] * pattern[j] / self.n_neurons

    def retrieve(self, noisy_pattern: list[int], max_iter: int = 20) -> list[int]:
        state = list(noisy_pattern)
        for _ in range(max_iter):
            prev_state = list(state)
            for i in range(self.n_neurons):
                net_input = sum(self.weights[i][j] * state[j] for j in range(self.n_neurons))
                state[i] = 1 if net_input >= 0 else -1
            if state == prev_state:
                break
        return state

    def energy(self, state: list[int]) -> float:
        e = 0.0
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                e += self.weights[i][j] * state[i] * state[j]
        return -0.5 * e

    @property
    def capacity(self) -> float:
        return 0.14 * self.n_neurons

if __name__ == "__main__":
    print("Agent Memory System Demo")
    agent = AgentMemorySystem(working_capacity=3)
    
    agent.perceive("User: Hi, my name is Alice and I like blue.")
    agent.remember("User's name is Alice.", importance=0.9)
    agent.remember("User's favorite color is blue.", importance=0.8)
    
    print(agent.build_context("What color do I like?"))

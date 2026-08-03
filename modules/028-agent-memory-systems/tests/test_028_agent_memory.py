import pytest
import math
import time
import os
import sys
import importlib.util

# Add parent dir to path to import 04-implementation.py
def load_impl():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "04-implementation.py")
    spec = importlib.util.spec_from_file_location("module_028_impl", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module_028_impl"] = module
    spec.loader.exec_module(module)
    return module

try:
    impl = load_impl()
except Exception:
    pytest.skip("04-implementation.py not found", allow_module_level=True)


# --- Category 1: Working Memory (4) ---

def test_working_memory_capacity_limit():
    wm = impl.WorkingMemory(capacity=3)
    wm.add("1")
    wm.add("2")
    wm.add("3")
    assert wm.is_full() == True
    wm.add("4")
    assert len(wm.get_all()) == 3

def test_working_memory_fifo_eviction():
    wm = impl.WorkingMemory(capacity=3)
    wm.add("1")
    wm.add("2")
    wm.add("3")
    wm.add("4")
    assert wm.get_all() == ["2", "3", "4"]

def test_working_memory_get_all_returns_contents():
    wm = impl.WorkingMemory(capacity=5)
    wm.add("A")
    wm.add("B")
    assert wm.get_all() == ["A", "B"]

def test_working_memory_clear():
    wm = impl.WorkingMemory(capacity=5)
    wm.add("A")
    wm.clear()
    assert len(wm.get_all()) == 0


# --- Category 2: Long-Term Memory Store (4) ---

@pytest.fixture
def memory_store():
    embedder = impl.EmbeddingModel(vocab_size=100)
    return impl.MemoryStore(embedder=embedder)

def test_memory_store_stores_memory(memory_store):
    mem = memory_store.store("Test content")
    assert mem.id in memory_store.memories
    assert mem.content == "Test content"

def test_memory_store_retrieve_by_similarity(memory_store):
    memory_store.store("The color of the sky is blue")
    memory_store.store("Dogs are good pets")
    
    res = memory_store.retrieve("sky color", k=1)
    assert len(res) == 1
    assert "sky" in res[0].content.lower()

def test_memory_store_update_content(memory_store):
    mem = memory_store.store("Old content")
    memory_store.update(mem.id, "New content")
    assert memory_store.memories[mem.id].content == "New content"

def test_memory_store_delete(memory_store):
    mem = memory_store.store("To be deleted")
    res = memory_store.delete(mem.id)
    assert res == True
    assert mem.id not in memory_store.memories


# --- Category 3: Agent Memory System (4) ---

@pytest.fixture
def agent_system():
    return impl.AgentMemorySystem(working_capacity=2, long_term_capacity=100)

def test_agent_memory_perceive_adds_to_working(agent_system):
    agent_system.perceive("Observation 1")
    assert len(agent_system.working_memory.get_all()) == 1
    assert agent_system.working_memory.get_all()[0] == "Observation 1"

def test_agent_memory_recall_returns_relevant(agent_system):
    agent_system.remember("Alice likes apples")
    agent_system.remember("Bob likes bananas")
    res = agent_system.recall("What does Alice like?", k=1)
    assert "Alice" in res[0].content

def test_agent_memory_build_context_includes_memories(agent_system):
    agent_system.perceive("Current chat")
    agent_system.remember("Past fact")
    context = agent_system.build_context("fact")
    assert "Current chat" in context
    assert "Past fact" in context

def test_agent_memory_consolidate_prunes_old(agent_system):
    # Store a memory and artificially age it
    mem = agent_system.remember("Old fact", importance=0.1)
    # Move timestamp back by 100 days
    mem.timestamp -= 100 * 24 * 3600
    
    pruned = agent_system.long_term_memory.consolidate()
    assert pruned == 1
    assert mem.id not in agent_system.long_term_memory.memories


# --- Category 4: Mathematical Properties (4) ---

def test_cosine_similarity_identical_vectors():
    embedder = impl.EmbeddingModel()
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert math.isclose(embedder.cosine_similarity(v1, v2), 1.0)

def test_cosine_similarity_orthogonal_vectors():
    embedder = impl.EmbeddingModel()
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    assert math.isclose(embedder.cosine_similarity(v1, v2), 0.0)

def test_forgetting_curve_decreases_over_time():
    embedder = impl.EmbeddingModel()
    store = impl.MemoryStore(embedder=embedder)
    ret_t1 = store.forget_curve(elapsed_hours=10, strength=2.0)
    ret_t2 = store.forget_curve(elapsed_hours=20, strength=2.0)
    assert ret_t2 < ret_t1

def test_hopfield_retrieves_stored_pattern():
    hn = impl.HopfieldNetwork(n_neurons=4)
    p = [1, -1, 1, -1]
    hn.store_pattern(p)
    noisy = [-1, -1, 1, -1] # one bit flipped
    recovered = hn.retrieve(noisy)
    assert recovered == p

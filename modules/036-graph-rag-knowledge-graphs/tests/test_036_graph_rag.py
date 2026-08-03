import pytest
import os
import importlib.util

def load_module():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    impl_path = os.path.join(current_dir, '..', '04-implementation.py')
    spec = importlib.util.spec_from_file_location("module_036_impl", impl_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

impl = load_module()

# Category 1: Knowledge Graph Operations (4)
def test_kg_add_entity():
    kg = impl.KnowledgeGraph()
    e = impl.Entity("e1", "Person", {"name": "Alice"})
    kg.add_entity(e)
    assert len(kg.entities) == 1
    assert kg.entities["e1"].type == "Person"

def test_kg_add_relation():
    kg = impl.KnowledgeGraph()
    kg.add_entity(impl.Entity("e1", "Person"))
    kg.add_entity(impl.Entity("e2", "Company"))
    kg.add_relation(impl.Relation("e1", "e2", "WORKS_AT", 0.9))
    assert len(kg.relations) == 1
    assert kg.get_neighbors("e1") == {"e2": 0.9}

def test_kg_relation_missing_entity():
    kg = impl.KnowledgeGraph()
    with pytest.raises(ValueError):
        kg.add_relation(impl.Relation("e1", "e2", "KNOWS"))

def test_kg_get_nodes():
    kg = impl.KnowledgeGraph()
    kg.add_entity(impl.Entity("e1", "Person"))
    kg.add_entity(impl.Entity("e2", "Person"))
    assert set(kg.get_nodes()) == {"e1", "e2"}

# Category 2: Graph Extraction & Parsing (4)
def test_extractor_initialization():
    ex = impl.GraphExtractor()
    assert ex.mock_responses == {}

def test_extractor_mock_extraction():
    ex = impl.GraphExtractor()
    e1 = impl.Entity("e1", "Person")
    r1 = impl.Relation("e1", "e2", "KNOWS")
    ex.add_mock_response("text", [e1], [r1])
    e, r = ex.extract("text")
    assert len(e) == 1
    assert len(r) == 1

def test_extractor_empty_extraction():
    ex = impl.GraphExtractor()
    e, r = ex.extract("unknown text")
    assert len(e) == 0
    assert len(r) == 0

def test_extractor_multiple_texts():
    ex = impl.GraphExtractor()
    ex.add_mock_response("t1", [impl.Entity("e1", "T")], [])
    ex.add_mock_response("t2", [impl.Entity("e2", "T")], [])
    assert len(ex.extract("t1")[0]) == 1
    assert ex.extract("t1")[0][0].id == "e1"
    assert len(ex.extract("t2")[0]) == 1
    assert ex.extract("t2")[0][0].id == "e2"

# Category 3: PageRank & Graph Search (4)
def test_pagerank_empty():
    pr = impl.PageRankRanker()
    kg = impl.KnowledgeGraph()
    assert pr.rank(kg) == {}

def test_pagerank_simple():
    pr = impl.PageRankRanker(damping=0.85, max_iter=10)
    kg = impl.KnowledgeGraph()
    kg.add_entity(impl.Entity("A", "N"))
    kg.add_entity(impl.Entity("B", "N"))
    kg.add_relation(impl.Relation("A", "B", "R", 1.0))
    kg.add_relation(impl.Relation("B", "A", "R", 1.0))
    ranks = pr.rank(kg)
    assert abs(ranks["A"] - 0.5) < 0.05
    assert abs(ranks["B"] - 0.5) < 0.05

def test_retriever_global_search():
    kg = impl.KnowledgeGraph()
    kg.add_entity(impl.Entity("A", "N"))
    kg.add_entity(impl.Entity("B", "N"))
    kg.add_relation(impl.Relation("A", "B", "R", 1.0))
    retriever = impl.GraphRAGRetriever(kg)
    res = retriever.global_search()
    assert len(res) == 2

def test_retriever_local_search():
    kg = impl.KnowledgeGraph()
    kg.add_entity(impl.Entity("A", "N"))
    kg.add_entity(impl.Entity("B", "N"))
    kg.add_entity(impl.Entity("C", "N"))
    kg.add_relation(impl.Relation("A", "B", "R", 0.5))
    kg.add_relation(impl.Relation("B", "C", "R", 0.5))
    retriever = impl.GraphRAGRetriever(kg)
    res = retriever.local_search("A", hops=2)
    assert "B" in res
    assert "C" in res
    assert res["C"] == 0.25

# Category 4: Community Detection (4)
def test_community_empty():
    cd = impl.LeidenCommunityDetector()
    kg = impl.KnowledgeGraph()
    assert cd.detect(kg) == {}

def test_community_disconnected():
    cd = impl.LeidenCommunityDetector()
    kg = impl.KnowledgeGraph()
    kg.add_entity(impl.Entity("A", "N"))
    kg.add_entity(impl.Entity("B", "N"))
    comms = cd.detect(kg)
    assert comms["A"] != comms["B"]

def test_community_connected():
    cd = impl.LeidenCommunityDetector()
    kg = impl.KnowledgeGraph()
    kg.add_entity(impl.Entity("A", "N"))
    kg.add_entity(impl.Entity("B", "N"))
    kg.add_relation(impl.Relation("A", "B", "R", 1.0))
    kg.add_relation(impl.Relation("B", "A", "R", 1.0))
    comms = cd.detect(kg)
    assert comms["A"] == comms["B"]

def test_community_two_components():
    cd = impl.LeidenCommunityDetector()
    kg = impl.KnowledgeGraph()
    kg.add_entity(impl.Entity("A", "N"))
    kg.add_entity(impl.Entity("B", "N"))
    kg.add_entity(impl.Entity("C", "N"))
    kg.add_entity(impl.Entity("D", "N"))
    kg.add_relation(impl.Relation("A", "B", "R", 1.0))
    kg.add_relation(impl.Relation("B", "A", "R", 1.0))
    kg.add_relation(impl.Relation("C", "D", "R", 1.0))
    kg.add_relation(impl.Relation("D", "C", "R", 1.0))
    comms = cd.detect(kg)
    assert comms["A"] == comms["B"]
    assert comms["C"] == comms["D"]
    assert comms["A"] != comms["C"]

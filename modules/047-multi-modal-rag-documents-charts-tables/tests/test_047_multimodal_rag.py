import pytest
import os
import sys
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def impl():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    impl_path = os.path.join(current_dir, "..", "04-implementation.py")
    return load_module("module_047_impl", impl_path)

# Bounding Box Math & IoU (4)
def test_bbox_area(impl):
    box = impl.BoundingBox(0, 0, 10, 10)
    assert box.area() == 100

def test_bbox_intersection(impl):
    box1 = impl.BoundingBox(0, 0, 10, 10)
    box2 = impl.BoundingBox(5, 5, 15, 15)
    inter = box1.intersection(box2)
    assert inter.area() == 25
    assert inter.x0 == 5
    assert inter.y0 == 5
    assert inter.x1 == 10
    assert inter.y1 == 10

def test_bbox_no_intersection(impl):
    box1 = impl.BoundingBox(0, 0, 10, 10)
    box2 = impl.BoundingBox(20, 20, 30, 30)
    inter = box1.intersection(box2)
    assert inter is None

def test_bbox_iou(impl):
    box1 = impl.BoundingBox(0, 0, 10, 10)
    box2 = impl.BoundingBox(5, 5, 15, 15)
    iou = box1.iou(box2)
    assert abs(iou - 25/175) < 1e-6

# Table Structure Parsing (4)
def test_table_parser_empty(impl):
    parser = impl.TableStructureParser()
    assert parser.parse_grid_to_csv([]) == ""

def test_table_parser_single_cell(impl):
    parser = impl.TableStructureParser()
    csv = parser.parse_grid_to_csv([{'row': 0, 'col': 0, 'text': 'Hello'}])
    assert csv == "Hello"

def test_table_parser_grid(impl):
    parser = impl.TableStructureParser()
    cells = [
        {'row': 0, 'col': 0, 'text': 'Name'}, {'row': 0, 'col': 1, 'text': 'Age'},
        {'row': 1, 'col': 0, 'text': 'Alice'}, {'row': 1, 'col': 1, 'text': '30'}
    ]
    csv = parser.parse_grid_to_csv(cells)
    assert csv == "Name,Age\nAlice,30"

def test_table_parser_missing_cells(impl):
    parser = impl.TableStructureParser()
    cells = [
        {'row': 0, 'col': 0, 'text': 'A'},
        {'row': 1, 'col': 1, 'text': 'B'}
    ]
    csv = parser.parse_grid_to_csv(cells)
    assert csv == "A,\n,B"

# Layout-Aware Chunker (4)
def test_chunker_empty(impl):
    chunker = impl.LayoutAwareChunker()
    assert chunker.chunk([]) == []

def test_chunker_text_order(impl):
    chunker = impl.LayoutAwareChunker()
    b1 = impl.DocumentBlock('text', 'Bottom', impl.BoundingBox(0, 100, 10, 110))
    b2 = impl.DocumentBlock('text', 'Top', impl.BoundingBox(0, 10, 10, 20))
    chunks = chunker.chunk([b1, b2])
    assert chunks == ["Top\nBottom"]

def test_chunker_table_isolation(impl):
    chunker = impl.LayoutAwareChunker()
    b1 = impl.DocumentBlock('text', 'Intro', impl.BoundingBox(0, 0, 10, 10))
    b2 = impl.DocumentBlock('table', 'A,B\n1,2', impl.BoundingBox(0, 20, 10, 30))
    b3 = impl.DocumentBlock('text', 'Outro', impl.BoundingBox(0, 40, 10, 50))
    chunks = chunker.chunk([b1, b2, b3])
    assert len(chunks) == 3
    assert chunks[0] == "Intro"
    assert chunks[1] == "[TABLE]\nA,B\n1,2"
    assert chunks[2] == "Outro"

def test_chunker_header_formatting(impl):
    chunker = impl.LayoutAwareChunker()
    b1 = impl.DocumentBlock('header', 'Title', impl.BoundingBox(0, 0, 10, 10))
    chunks = chunker.chunk([b1])
    assert chunks == ["# Title"]

# Multi-Modal Retrieval (4)
def test_retriever_empty(impl):
    r = impl.MultiModalDocRetriever([])
    assert r.search("query") == []

def test_retriever_exact_match(impl):
    r = impl.MultiModalDocRetriever(["apple pie", "banana split"])
    assert r.search("apple") == ["apple pie"]

def test_retriever_ranking(impl):
    r = impl.MultiModalDocRetriever(["apple pie", "apple apple tree", "banana split"])
    res = r.search("apple")
    assert res[0] == "apple apple tree"
    assert res[1] == "apple pie"
    assert len(res) == 2

def test_retriever_table_match(impl):
    r = impl.MultiModalDocRetriever(["Intro", "[TABLE]\nRevenue,100M", "Outro"])
    res = r.search("revenue")
    assert res == ["[TABLE]\nRevenue,100M"]

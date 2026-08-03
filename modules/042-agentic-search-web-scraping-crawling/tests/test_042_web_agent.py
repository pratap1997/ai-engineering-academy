import pytest
import importlib.util
import sys
import os

# Dynamically import implementation
impl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "04-implementation.py"))
spec = importlib.util.spec_from_file_location("module_042_impl", impl_path)
impl = importlib.util.module_from_spec(spec)
sys.modules["module_042_impl"] = impl
spec.loader.exec_module(impl)

# Category 1: HTML Parser & Markdown Conversion (4 tests)
def test_html_parser_strips_script():
    parser = impl.HTMLCleanerParser()
    parser.feed("<html><script>alert('x');</script><body>hello</body></html>")
    assert "hello" in parser.get_markdown()
    assert "alert" not in parser.get_markdown()

def test_html_parser_strips_style():
    parser = impl.HTMLCleanerParser()
    parser.feed("<style>body {color: red;}</style>text")
    assert "text" in parser.get_markdown()
    assert "color" not in parser.get_markdown()

def test_html_parser_basic_text():
    parser = impl.HTMLCleanerParser()
    parser.feed("<p>Hello World</p>")
    assert parser.get_markdown() == "Hello World"
    
def test_html_parser_multiple_tags():
    parser = impl.HTMLCleanerParser()
    parser.feed("<div><p>A</p><p>B</p></div>")
    assert "A" in parser.get_markdown() and "B" in parser.get_markdown()

# Category 2: Self-Healing Locators (4 tests)
def test_fingerprint_match():
    fp = impl.DOMElementFingerprinter("target")
    assert fp.match("<div>target</div>") == True

def test_fingerprint_no_match():
    fp = impl.DOMElementFingerprinter("target")
    assert fp.match("<div>other</div>") == False
    
def test_fingerprint_init():
    fp = impl.DOMElementFingerprinter("text")
    assert fp.target_text == "text"

def test_fingerprint_complex_html():
    fp = impl.DOMElementFingerprinter("content")
    assert fp.match("<div class='x'><span>content</span></div>") == True

# Category 3: Crawl Graph Traversal (4 tests)
def test_crawl_depth_0():
    crawler = impl.CrawlGraphEngine(max_depth=0)
    res = crawler.crawl("http://a.com")
    assert len(res) == 1
    
def test_crawl_depth_1():
    crawler = impl.CrawlGraphEngine(max_depth=1)
    res = crawler.crawl("http://a.com")
    assert len(res) == 2

def test_crawl_visited():
    crawler = impl.CrawlGraphEngine(max_depth=1)
    crawler.crawl("http://a.com")
    assert len(crawler.visited) == 2

def test_crawl_results_contain_start():
    crawler = impl.CrawlGraphEngine(max_depth=2)
    res = crawler.crawl("http://start.com")
    assert res[0] == "http://start.com"

# Category 4: Data Extraction (4 tests)
def test_extractor_init():
    ext = impl.WebDataExtractor({"title": "string"})
    assert "title" in ext.schema
    
def test_extractor_extracts_keys():
    ext = impl.WebDataExtractor({"title": "string"})
    res = ext.extract("html")
    assert "title" in res

def test_extractor_mock_value():
    ext = impl.WebDataExtractor({"name": "string"})
    res = ext.extract("html")
    assert res["name"] == "Extracted name"
    
def test_extractor_multiple_keys():
    ext = impl.WebDataExtractor({"a": "1", "b": "2"})
    res = ext.extract("html")
    assert len(res) == 2

import pytest
import importlib.util
import os

def load_impl():
    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    impl_path = os.path.join(dir_path, "04-implementation.py")
    spec = importlib.util.spec_from_file_location("module_049_impl", impl_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

impl = load_impl()

# --- Prompt Compression (4) ---
def test_compression_removes_whitespace():
    compressor = impl.PromptCompressor()
    assert compressor.compress("hello    world") == "hello world"

def test_compression_removes_stopwords():
    compressor = impl.PromptCompressor()
    assert compressor.compress("please do this") == "this"

def test_compression_custom_stopwords():
    compressor = impl.PromptCompressor(stop_words=["custom"])
    assert compressor.compress("custom word") == "word"

def test_compression_empty_string():
    compressor = impl.PromptCompressor()
    assert compressor.compress("   ") == ""

# --- Caching Layers (4) ---
def test_exact_cache_hit():
    cache = impl.ExactMatchCache()
    cache.set("key", "val")
    assert cache.get("key") == "val"

def test_exact_cache_miss():
    cache = impl.ExactMatchCache()
    assert cache.get("key") is None

def test_semantic_cache_hit():
    cache = impl.SemanticVectorCache(similarity_threshold=0.4)
    cache.set("machine learning", "AI")
    assert cache.get("what is machine learning") == "AI"

def test_semantic_cache_miss():
    cache = impl.SemanticVectorCache(similarity_threshold=0.9)
    cache.set("machine learning", "AI")
    assert cache.get("how to cook pasta") is None

# --- Provider Failover Routing (4) ---
def test_router_success_first():
    providers = [{"name": "P1", "func": lambda x: "R1"}]
    router = impl.ProviderFailoverRouter(providers)
    assert router.route("Q") == "R1"

def test_router_failover():
    def fail(x):
        raise ValueError("Fail")
    providers = [
        {"name": "P1", "func": fail},
        {"name": "P2", "func": lambda x: "R2"}
    ]
    router = impl.ProviderFailoverRouter(providers)
    assert router.route("Q") == "R2"
    assert not router.providers[0]["is_healthy"]

def test_router_all_fail():
    def fail(x):
        raise ValueError("Fail")
    providers = [{"name": "P1", "func": fail}]
    router = impl.ProviderFailoverRouter(providers)
    with pytest.raises(RuntimeError):
        router.route("Q")

def test_router_skips_unhealthy():
    def fail(x):
        raise ValueError("Fail")
    providers = [
        {"name": "P1", "func": fail, "is_healthy": False},
        {"name": "P2", "func": lambda x: "R2"}
    ]
    router = impl.ProviderFailoverRouter(providers)
    assert router.route("Q") == "R2"

# --- Gateway Proxy Operations (4) ---
def test_gateway_cache_miss_routes():
    router = impl.ProviderFailoverRouter([{"name": "P1", "func": lambda x: f"R({x})"}])
    engine = impl.AIGatewayEngine(router)
    res = engine.process("please route this")
    assert res == "R(route this)"

def test_gateway_exact_cache_hit():
    router = impl.ProviderFailoverRouter([{"name": "P1", "func": lambda x: "Should Not Call"}])
    engine = impl.AIGatewayEngine(router)
    engine.exact_cache.set("q", "cached")
    assert engine.process("q") == "cached"

def test_gateway_semantic_cache_hit():
    router = impl.ProviderFailoverRouter([{"name": "P1", "func": lambda x: "Should Not Call"}])
    engine = impl.AIGatewayEngine(router)
    engine.semantic_cache.set("q", "sem_cached")
    assert engine.process("please q") == "sem_cached"
    
def test_gateway_caches_response():
    router = impl.ProviderFailoverRouter([{"name": "P1", "func": lambda x: "API"}])
    engine = impl.AIGatewayEngine(router)
    engine.process("fresh")
    assert engine.exact_cache.get("fresh") == "API"

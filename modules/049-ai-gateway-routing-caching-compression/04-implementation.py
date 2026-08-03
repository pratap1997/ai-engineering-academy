import time
import math
import re
from typing import List, Dict, Any, Optional, Tuple

class PromptCompressor:
    def __init__(self, stop_words: Optional[List[str]] = None):
        if stop_words is None:
            self.stop_words = ["please", "could", "you", "the", "a", "an", "is", "are", "to", "do", "of"]
        else:
            self.stop_words = stop_words
            
    def compress(self, text: str) -> str:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove stop words (case insensitive) for simulation
        words = text.split()
        compressed_words = [w for w in words if w.lower() not in self.stop_words]
        return " ".join(compressed_words)

class ExactMatchCache:
    def __init__(self):
        self._cache: Dict[str, str] = {}
        
    def get(self, key: str) -> Optional[str]:
        return self._cache.get(key)
        
    def set(self, key: str, value: str):
        self._cache[key] = value
        
    def clear(self):
        self._cache.clear()

class SemanticVectorCache:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        # Stores tuples of (text, response)
        self._cache: List[Tuple[str, str]] = []
        
    def _get_bow(self, text: str) -> Dict[str, int]:
        words = text.lower().split()
        bow = {}
        for w in words:
            bow[w] = bow.get(w, 0) + 1
        return bow
        
    def _cosine_similarity(self, bow1: Dict[str, int], bow2: Dict[str, int]) -> float:
        intersection = set(bow1.keys()) & set(bow2.keys())
        dot_product = sum(bow1[w] * bow2[w] for w in intersection)
        
        mag1 = math.sqrt(sum(v**2 for v in bow1.values()))
        mag2 = math.sqrt(sum(v**2 for v in bow2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
            
        return dot_product / (mag1 * mag2)
        
    def get(self, query: str) -> Optional[str]:
        query_bow = self._get_bow(query)
        best_match = None
        highest_sim = 0.0
        
        for cached_query, response in self._cache:
            sim = self._cosine_similarity(query_bow, self._get_bow(cached_query))
            if sim > highest_sim:
                highest_sim = sim
                best_match = response
                
        if highest_sim >= self.similarity_threshold:
            return best_match
        return None
        
    def set(self, query: str, response: str):
        self._cache.append((query, response))
        
    def clear(self):
        self._cache.clear()

class ProviderFailoverRouter:
    def __init__(self, providers: List[Dict[str, Any]]):
        # provider dict: {"name": str, "func": callable, "is_healthy": bool, "latency_ms": int}
        self.providers = providers
        for p in self.providers:
            if "is_healthy" not in p:
                p["is_healthy"] = True
                
    def get_next_healthy(self) -> Optional[Dict[str, Any]]:
        # For simplicity, just return the first healthy one
        for p in self.providers:
            if p["is_healthy"]:
                return p
        return None
        
    def mark_unhealthy(self, name: str):
        for p in self.providers:
            if p["name"] == name:
                p["is_healthy"] = False
                
    def mark_healthy(self, name: str):
        for p in self.providers:
            if p["name"] == name:
                p["is_healthy"] = True

    def route(self, prompt: str) -> str:
        # Tries to execute using providers in order
        for provider in self.providers:
            if not provider["is_healthy"]:
                continue
            try:
                # simulate calling
                res = provider["func"](prompt)
                return res
            except Exception:
                self.mark_unhealthy(provider["name"])
        raise RuntimeError("All providers failed")

class AIGatewayEngine:
    def __init__(self, router: ProviderFailoverRouter):
        self.compressor = PromptCompressor()
        self.exact_cache = ExactMatchCache()
        self.semantic_cache = SemanticVectorCache(similarity_threshold=0.8)
        self.router = router
        
    def process(self, prompt: str) -> str:
        # 1. Exact match
        exact_res = self.exact_cache.get(prompt)
        if exact_res:
            return exact_res
            
        # 2. Compress prompt
        compressed_prompt = self.compressor.compress(prompt)
        
        # 3. Semantic match (using compressed to be more robust)
        sem_res = self.semantic_cache.get(compressed_prompt)
        if sem_res:
            return sem_res
            
        # 4. Route
        response = self.router.route(compressed_prompt)
        
        # 5. Cache for future
        self.exact_cache.set(prompt, response)
        self.semantic_cache.set(compressed_prompt, response)
        
        return response

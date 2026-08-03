import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class RAGQuery:
    """A query through the RAG pipeline."""
    text: str
    query_type: str  # "simple", "complex", "multi-hop", "clarification"
    metadata: dict = field(default_factory=dict)

@dataclass
class RAGResult:
    """Result from a RAG call with full trace."""
    query: RAGQuery
    answer: str
    chunks: List[Dict[str, Any]]
    latency_ms: float
    cache_hit: bool
    trace: List[Dict[str, Any]]

class QueryClassifier:
    """Classify query complexity to route to appropriate retrieval strategy."""
    def classify(self, query: str) -> str:
        # Mock logic based on keywords or length
        lower_query = query.lower()
        if "compare" in lower_query or "difference" in lower_query or "and" in lower_query:
            return "complex"
        if len(query.split()) > 15:
            return "complex"
        return "simple"

class RAGCache:
    """Two-level cache: exact match (query) + semantic (embedding similarity)."""
    def __init__(self, exact_capacity: int = 100, semantic_capacity: int = 500):
        self.exact_capacity = exact_capacity
        self.semantic_capacity = semantic_capacity
        self.exact_cache: Dict[str, RAGResult] = {}
        self.semantic_cache: List[tuple] = []  # List of (embedding, query, result)
        self.hits = 0
        self.misses = 0

    def _mock_embed(self, text: str) -> List[float]:
        # Simple deterministic mock embedding based on length
        return [float(len(text)), float(len(text.split()))]

    def _cosine_sim(self, v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def get(self, query: str) -> Optional[RAGResult]:
        # Check exact
        if query in self.exact_cache:
            self.hits += 1
            return self.exact_cache[query]
        
        # Check semantic
        q_emb = self._mock_embed(query)
        for emb, orig_query, result in self.semantic_cache:
            if self._cosine_sim(q_emb, emb) > 0.95:  # High threshold for semantic match
                self.hits += 1
                return result
                
        self.misses += 1
        return None

    def put(self, query: str, result: RAGResult) -> None:
        if len(self.exact_cache) >= self.exact_capacity:
            self.evict_lru()
        self.exact_cache[query] = result
        
        if len(self.semantic_cache) >= self.semantic_capacity:
            self.semantic_cache.pop(0)  # Simple FIFO for semantic cache
        self.semantic_cache.append((self._mock_embed(query), query, result))

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def evict_lru(self) -> None:
        if self.exact_cache:
            # Simple eviction: remove arbitrary item
            first_key = next(iter(self.exact_cache))
            del self.exact_cache[first_key]

class IndexManager:
    """Manages document indices with incremental updates and staleness detection."""
    def __init__(self):
        self.documents: Dict[str, str] = {}
        self.changes_since_rebuild = 0
        self.threshold = 0.1  # 10%

    def add_document(self, doc_id: str, content: str) -> None:
        if doc_id not in self.documents:
            self.documents[doc_id] = content
            self.changes_since_rebuild += 1

    def update_document(self, doc_id: str, content: str) -> None:
        if doc_id in self.documents and self.documents[doc_id] != content:
            self.documents[doc_id] = content
            self.changes_since_rebuild += 1

    def delete_document(self, doc_id: str) -> None:
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.changes_since_rebuild += 1

    def staleness_score(self) -> float:
        total = len(self.documents)
        return self.changes_since_rebuild / total if total > 0 else 0.0

    def rebuild_needed(self) -> bool:
        return self.staleness_score() >= self.threshold
        
    def rebuild(self):
        self.changes_since_rebuild = 0

class RAGASEvaluator:
    """RAGAS-style evaluation for RAG pipeline quality."""
    def faithfulness(self, answer: str, context_chunks: List[str]) -> float:
        # Mock logic: checks how many words in answer are in context
        ans_words = set(answer.lower().split())
        ctx_words = set(" ".join(context_chunks).lower().split())
        if not ans_words:
            return 1.0
        supported = sum(1 for w in ans_words if w in ctx_words)
        return supported / len(ans_words)

    def answer_relevancy(self, question: str, answer: str) -> float:
        # Mock logic: compares token overlap
        q_words = set(question.lower().split())
        ans_words = set(answer.lower().split())
        if not q_words:
            return 1.0
        overlap = len(q_words.intersection(ans_words))
        return min(1.0, overlap / max(1, len(q_words) * 0.5))

    def context_precision(self, question: str, retrieved_chunks: List[str], relevant_chunks: List[str]) -> float:
        if not relevant_chunks or not retrieved_chunks:
            return 0.0
        # Precision@K style mock
        relevant_set = set(relevant_chunks)
        score = 0.0
        hits = 0
        for k, chunk in enumerate(retrieved_chunks):
            if chunk in relevant_set:
                hits += 1
                score += hits / (k + 1)
        return score / min(len(relevant_set), len(retrieved_chunks))

    def context_recall(self, retrieved_chunks: List[str], ground_truth_chunks: List[str]) -> float:
        if not ground_truth_chunks:
            return 1.0
        retrieved_set = set(retrieved_chunks)
        hits = sum(1 for gt in ground_truth_chunks if gt in retrieved_set)
        return hits / len(ground_truth_chunks)

    def full_evaluation(self, question: str, answer: str, retrieved: List[str], ground_truth: List[str]) -> Dict[str, float]:
        return {
            "faithfulness": self.faithfulness(answer, retrieved),
            "answer_relevancy": self.answer_relevancy(question, answer),
            "context_precision": self.context_precision(question, retrieved, ground_truth),
            "context_recall": self.context_recall(retrieved, ground_truth),
        }

class LatencyBudget:
    """Enforce per-stage latency budgets across the RAG pipeline."""
    def __init__(self, total_ms: float = 500.0):
        self.total_ms = total_ms
        self.allocations: Dict[str, float] = {}
        self.actuals: Dict[str, float] = {}

    def allocate(self, stage: str, allocated_ms: float) -> None:
        self.allocations[stage] = allocated_ms

    def record(self, stage: str, actual_ms: float) -> None:
        self.actuals[stage] = actual_ms

    def check(self, stage: str, actual_ms: float) -> bool:
        self.record(stage, actual_ms)
        return actual_ms <= self.allocations.get(stage, float('inf'))

    def report(self) -> Dict[str, Any]:
        return {
            "total_budget": self.total_ms,
            "total_actual": sum(self.actuals.values()),
            "within_budget": sum(self.actuals.values()) <= self.total_ms,
            "stages": {s: {"budget": self.allocations.get(s, 0), "actual": self.actuals.get(s, 0)} for s in self.actuals}
        }

class MockRetriever:
    def retrieve(self, query: str, is_complex: bool) -> List[Dict[str, Any]]:
        # Mock retrieval with latency
        time.sleep(0.1) # Simulate 100ms
        if is_complex:
            time.sleep(0.1) # Additional 100ms for reranking
        
        # Deterministic mock context
        return [{"id": "doc1", "content": f"Information about {query}"}, 
                {"id": "doc2", "content": f"More details regarding {query}"}]

class MockGenerator:
    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        # Mock generation with latency
        time.sleep(0.15) # Simulate 150ms
        ctx_str = " ".join([c["content"] for c in context])
        return f"Based on {ctx_str}, the answer to '{query}' is clear."

class ProductionRAGPipeline:
    """End-to-end production RAG pipeline with all components."""
    def __init__(self, retriever, generator, cache: RAGCache, evaluator: RAGASEvaluator, budget: LatencyBudget):
        self.retriever = retriever
        self.generator = generator
        self.cache = cache
        self.evaluator = evaluator
        self.budget = budget
        self.classifier = QueryClassifier()

    def query(self, question: str) -> RAGResult:
        start_time = time.time()
        trace = []
        
        # 1. Check Cache
        c_start = time.time()
        cached = self.cache.get(question)
        c_latency = (time.time() - c_start) * 1000
        self.budget.record("cache", c_latency)
        
        if cached:
            cached.latency_ms = (time.time() - start_time) * 1000
            cached.trace.append({"stage": "cache", "latency": c_latency, "status": "hit"})
            return cached
            
        trace.append({"stage": "cache", "latency": c_latency, "status": "miss"})

        # 2. Classify Query
        cl_start = time.time()
        q_type = self.classifier.classify(question)
        rag_query = RAGQuery(text=question, query_type=q_type)
        cl_latency = (time.time() - cl_start) * 1000
        self.budget.record("classification", cl_latency)
        trace.append({"stage": "classification", "latency": cl_latency, "result": q_type})

        # 3. Retrieve
        r_start = time.time()
        chunks = self.retriever.retrieve(question, is_complex=(q_type == "complex"))
        r_latency = (time.time() - r_start) * 1000
        self.budget.record("retrieval", r_latency)
        trace.append({"stage": "retrieval", "latency": r_latency, "num_chunks": len(chunks)})

        # 4. Generate
        g_start = time.time()
        answer = self.generator.generate(question, chunks)
        g_latency = (time.time() - g_start) * 1000
        self.budget.record("generation", g_latency)
        trace.append({"stage": "generation", "latency": g_latency})

        total_latency = (time.time() - start_time) * 1000
        
        result = RAGResult(
            query=rag_query,
            answer=answer,
            chunks=chunks,
            latency_ms=total_latency,
            cache_hit=False,
            trace=trace
        )
        
        # Save to cache
        self.cache.put(question, result)
        
        return result

    def batch_evaluate(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        for tc in test_cases:
            q = tc["question"]
            gt = tc["ground_truth"]
            res = self.query(q)
            retrieved_texts = [c["content"] for c in res.chunks]
            eval_scores = self.evaluator.full_evaluation(q, res.answer, retrieved_texts, gt)
            results.append({
                "question": q,
                "latency_ms": res.latency_ms,
                "scores": eval_scores
            })
            
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        avg_faithfulness = sum(r["scores"]["faithfulness"] for r in results) / len(results)
        
        return {
            "avg_latency": avg_latency,
            "avg_faithfulness": avg_faithfulness,
            "detailed": results
        }

    def health_report(self) -> Dict[str, Any]:
        return {
            "cache_hit_rate": self.cache.hit_rate(),
            "budget_report": self.budget.report()
        }

if __name__ == "__main__":
    retriever = MockRetriever()
    generator = MockGenerator()
    cache = RAGCache()
    evaluator = RAGASEvaluator()
    budget = LatencyBudget(500.0)
    budget.allocate("cache", 10.0)
    budget.allocate("classification", 10.0)
    budget.allocate("retrieval", 250.0)
    budget.allocate("generation", 200.0)
    
    pipeline = ProductionRAGPipeline(retriever, generator, cache, evaluator, budget)
    
    test_queries = [
        {"question": "What is RAG?", "ground_truth": ["Information about What is RAG?"]},
        {"question": "Compare RAG and Fine-tuning", "ground_truth": ["Information about Compare RAG and Fine-tuning"]}
    ]
    
    print("Running evaluation...")
    eval_res = pipeline.batch_evaluate(test_queries)
    print(f"Average Latency: {eval_res['avg_latency']:.2f} ms")
    print(f"Average Faithfulness: {eval_res['avg_faithfulness']:.2f}")
    print("\nHealth Report:")
    print(pipeline.health_report())

import math
from typing import List, Dict, Tuple, Optional

class BM25Scorer:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = []
        self.idf = {}
        self.avgdl = 0.0
        self.corpus_size = 0

    def fit(self, corpus: List[List[str]]):
        self.documents = corpus
        self.corpus_size = len(corpus)
        
        if self.corpus_size == 0:
            return
            
        total_len = 0
        df = {}
        
        for doc in corpus:
            total_len += len(doc)
            freq = {}
            for word in doc:
                freq[word] = freq.get(word, 0) + 1
            self.doc_freqs.append(freq)
            
            for word in freq.keys():
                df[word] = df.get(word, 0) + 1
                
        self.avgdl = total_len / self.corpus_size
        
        for word, freq in df.items():
            # IDF formula used in BM25
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def score(self, query: List[str], doc_idx: int) -> float:
        if self.corpus_size == 0 or doc_idx >= self.corpus_size:
            return 0.0
            
        doc_len = len(self.documents[doc_idx])
        doc_freqs = self.doc_freqs[doc_idx]
        
        score = 0.0
        for word in query:
            if word not in doc_freqs:
                continue
                
            f = doc_freqs[word]
            numerator = self.idf.get(word, 0.0) * f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            score += numerator / denominator
            
        return score
        
    def search(self, query: List[str]) -> List[Tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.corpus_size)]
        return sorted(scores, key=lambda x: x[1], reverse=True)


class DenseVectorRetriever:
    def __init__(self):
        self.vectors = []
        self.corpus_size = 0
        
    def fit(self, vectors: List[List[float]]):
        self.vectors = vectors
        self.corpus_size = len(vectors)
        
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
            
        dot_product = sum(x * y for x, y in zip(v1, v2))
        norm_v1 = math.sqrt(sum(x * x for x in v1))
        norm_v2 = math.sqrt(sum(x * x for x in v2))
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        return dot_product / (norm_v1 * norm_v2)
        
    def search(self, query_vector: List[float]) -> List[Tuple[int, float]]:
        scores = [(i, self._cosine_similarity(query_vector, self.vectors[i])) 
                  for i in range(self.corpus_size)]
        return sorted(scores, key=lambda x: x[1], reverse=True)


class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k
        
    def fuse(self, ranked_lists: List[List[Tuple[int, float]]]) -> List[Tuple[int, float]]:
        rrf_scores: Dict[int, float] = {}
        
        for ranked_list in ranked_lists:
            for rank, (doc_id, _) in enumerate(ranked_list):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (self.k + rank + 1)
                
        # Sort by RRF score descending
        fused = list(rrf_scores.items())
        return sorted(fused, key=lambda x: x[1], reverse=True)


class HybridSearchEngine:
    def __init__(self, bm25_scorer: BM25Scorer, dense_retriever: DenseVectorRetriever):
        self.bm25_scorer = bm25_scorer
        self.dense_retriever = dense_retriever
        
    def search_alpha_fusion(self, query_words: List[str], query_vector: List[float], alpha: float = 0.5) -> List[Tuple[int, float]]:
        bm25_results = self.bm25_scorer.search(query_words)
        dense_results = self.dense_retriever.search(query_vector)
        
        # Normalize BM25 scores (Min-Max normalization)
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results) if bm25_results else 1.0
            min_bm25 = min(score for _, score in bm25_results) if bm25_results else 0.0
            if max_bm25 > min_bm25:
                bm25_scores = {doc_id: (score - min_bm25) / (max_bm25 - min_bm25) for doc_id, score in bm25_results}
            else:
                bm25_scores = {doc_id: 1.0 if max_bm25 > 0 else 0.0 for doc_id, score in bm25_results}
        else:
            bm25_scores = {}
            
        # Normalize Dense scores (already between -1 and 1 usually, but let's 0-1 normalize if needed, though they are usually 0-1 for normalized vectors)
        if dense_results:
            max_dense = max(score for _, score in dense_results) if dense_results else 1.0
            min_dense = min(score for _, score in dense_results) if dense_results else 0.0
            if max_dense > min_dense:
                dense_scores = {doc_id: (score - min_dense) / (max_dense - min_dense) for doc_id, score in dense_results}
            else:
                dense_scores = {doc_id: 1.0 if max_dense > 0 else 0.0 for doc_id, score in dense_results}
        else:
            dense_scores = {}
            
        combined_scores = {}
        all_doc_ids = set(bm25_scores.keys()).union(set(dense_scores.keys()))
        
        for doc_id in all_doc_ids:
            s_sparse = bm25_scores.get(doc_id, 0.0)
            s_dense = dense_scores.get(doc_id, 0.0)
            combined_scores[doc_id] = alpha * s_dense + (1.0 - alpha) * s_sparse
            
        return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
    def search_rrf(self, query_words: List[str], query_vector: List[float], k: int = 60) -> List[Tuple[int, float]]:
        bm25_results = self.bm25_scorer.search(query_words)
        dense_results = self.dense_retriever.search(query_vector)
        
        rrf = ReciprocalRankFusion(k=k)
        return rrf.fuse([bm25_results, dense_results])

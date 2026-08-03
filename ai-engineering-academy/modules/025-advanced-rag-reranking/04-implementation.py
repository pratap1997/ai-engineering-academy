"""
AI ENGINEERING ACADEMY -- MODULE 025
Advanced RAG & Cross-Encoder Re-ranking Implementation (Pure Python & NumPy)

Provides:
1. `ParentChildChunker`: Hierarchical parent (context) to child (search index) document chunker.
2. `BiEncoderRetriever`: Fast candidate retrieval using cosine vector similarity.
3. `CrossEncoderReranker`: Fine-grained token cross-attention relevance scorer.
4. `AdvancedRAGPipeline`: End-to-end two-stage retrieval with parent chunk expansion.
"""

import numpy as np


def normalize_vectors(v):
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norms, 1e-12)


# =====================================================================
# 1. PARENT-CHILD CHUNKER
# =====================================================================

class ParentChildChunker:
    """Hierarchical text chunker creating small child chunks linked to large parent context chunks."""

    def __init__(self, parent_size=300, child_size=80, overlap=20):
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

    def chunk_document(self, text, doc_id="doc_0"):
        parents = []
        children = []

        # 1. Create Parent Chunks
        p_start = 0
        p_idx = 0
        while p_start < len(text):
            p_end = min(p_start + self.parent_size, len(text))
            parent_text = text[p_start:p_end]
            parent_id = f"{doc_id}_p{p_idx}"
            parents.append({"id": parent_id, "text": parent_text, "doc_id": doc_id})

            # 2. Create Child Chunks inside this Parent Chunk
            c_start = 0
            c_idx = 0
            while c_start < len(parent_text):
                c_end = min(c_start + self.child_size, len(parent_text))
                child_text = parent_text[c_start:c_end]
                child_id = f"{parent_id}_c{c_idx}"
                children.append({
                    "id": child_id,
                    "parent_id": parent_id,
                    "text": child_text,
                    "parent_text": parent_text
                })
                c_start += self.child_size - self.overlap
                c_idx += 1

            p_start += self.parent_size - self.overlap
            p_idx += 1

        return parents, children


# =====================================================================
# 2. BI-ENCODER RETRIEVER (STAGE 1: FAST RECALL)
# =====================================================================

class BiEncoderRetriever:
    """Simulated Bi-Encoder dense vector retrieval."""

    def __init__(self, dim=32):
        self.dim = dim
        self.child_chunks = []
        self.embeddings = None

    def index_chunks(self, children, seed=42):
        np.random.seed(seed)
        self.child_chunks = children
        N = len(children)
        # Generate random embeddings for simulation
        raw_emb = np.random.randn(N, self.dim)
        self.embeddings = normalize_vectors(raw_emb)

    def retrieve(self, query_emb, top_k=10):
        q = normalize_vectors(np.array(query_emb, dtype=np.float32))
        sims = np.dot(self.embeddings, q)
        top_indices = np.argsort(-sims)[:top_k]
        results = [self.child_chunks[i] for i in top_indices]
        scores = [float(sims[i]) for i in top_indices]
        return results, scores


# =====================================================================
# 3. CROSS-ENCODER RERANKER (STAGE 2: HIGH PRECISION)
# =====================================================================

class CrossEncoderReranker:
    """Simulated Cross-Encoder re-ranking with token cross-attention scoring."""

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.weights = np.random.randn(10)

    def score_pair(self, query_text, doc_text):
        """Computes query-document keyword overlap and token interaction score."""
        q_words = set(query_text.lower().split())
        d_words = set(doc_text.lower().split())

        overlap = len(q_words.intersection(d_words))
        len_ratio = min(len(q_words), len(d_words)) / max(1, max(len(q_words), len(d_words)))
        score = 0.8 * overlap + 0.2 * len_ratio
        return float(1.0 / (1.0 + np.exp(-score)))  # Sigmoid output

    def rerank(self, query_text, candidate_chunks, top_k=3):
        scored_candidates = []
        for chunk in candidate_chunks:
            score = self.score_pair(query_text, chunk["text"])
            scored_candidates.append((score, chunk))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        reranked_chunks = [item[1] for item in scored_candidates[:top_k]]
        reranked_scores = [item[0] for item in scored_candidates[:top_k]]

        return reranked_chunks, reranked_scores


# =====================================================================
# 4. ADVANCED RAG PIPELINE
# =====================================================================

class AdvancedRAGPipeline:
    """End-to-End Two-Stage RAG Pipeline with Parent Chunk Context Expansion."""

    def __init__(self, dim=32):
        self.chunker = ParentChildChunker()
        self.bi_encoder = BiEncoderRetriever(dim=dim)
        self.cross_encoder = CrossEncoderReranker()

    def ingest_document(self, text, doc_id="doc_1"):
        parents, children = self.chunker.chunk_document(text, doc_id)
        self.bi_encoder.index_chunks(children)
        return len(parents), len(children)

    def query(self, query_text, query_emb, stage1_k=10, final_k=3):
        # Stage 1: Fast Bi-Encoder Retrieval
        candidates, _ = self.bi_encoder.retrieve(query_emb, top_k=stage1_k)

        # Stage 2: Cross-Encoder Re-ranking
        reranked, scores = self.cross_encoder.rerank(query_text, candidates, top_k=final_k)

        # Stage 3: Parent Context Expansion
        parent_contexts = [chunk["parent_text"] for chunk in reranked]

        return parent_contexts, reranked, scores


def compute_precision_at_k(retrieved_ids, ground_truth_ids):
    overlap = len(set(retrieved_ids).intersection(set(ground_truth_ids)))
    return overlap / len(retrieved_ids)


def compute_map_at_k(retrieved_ids, ground_truth_ids):
    precisions = []
    hits = 0
    for i, item_id in enumerate(retrieved_ids):
        if item_id in ground_truth_ids:
            hits += 1
            precisions.append(hits / (i + 1))
    return np.mean(precisions) if precisions else 0.0


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 025 -- ADVANCED RAG & CROSS-ENCODER RE-RANKING VERIFICATION")
    print("=" * 65)

    sample_doc = """
    Artificial Intelligence is expanding rapidly across industries.
    Retrieval Augmented Generation combines dense vector retrieval with large language models.
    Bi-encoders encode queries and documents independently into vector space.
    Cross-encoders process query and document tokens jointly to compute deep cross-attention.
    Parent-child chunking pairs short searchable child chunks with long context parent chunks.
    """

    pipeline = AdvancedRAGPipeline(dim=16)
    n_p, n_c = pipeline.ingest_document(sample_doc, doc_id="doc_ai")

    print(f"\nDocument Chunking: {n_p} Parent Chunks, {n_c} Child Chunks Created")

    query_text = "Cross-encoders process query and document tokens jointly"
    query_emb = np.random.randn(16)

    contexts, chunks, scores = pipeline.query(query_text, query_emb, stage1_k=5, final_k=2)

    print("\n[RAG Retrieval Results]")
    for rank, (score, chunk, ctx) in enumerate(zip(scores, chunks, contexts)):
        print(f"  Rank {rank+1}: Score={score:.4f} | Child ID={chunk['id']}")
        print(f"    Parent Context: \"{ctx[:80].strip()}...\"")

    assert len(contexts) == 2
    assert scores[0] >= scores[1]
    print("\nAdvanced RAG Pipeline Verification Passed => [OK]")

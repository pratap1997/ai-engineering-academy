# 03 - Production RAG Systems: Mathematics

## RAGAS Metrics Formal Definitions

The RAGAS (Retrieval Augmented Generation Assessment) framework defines key metrics mathematically to evaluate RAG pipelines.

### 1. Faithfulness
Measures the hallucination rate. An answer is faithful if all claims it makes can be inferred from the retrieved context.

Let $A$ be the generated answer, and $C$ be the retrieved context. Let $S(A)$ be the set of individual claims extracted from $A$.

$$ \text{Faithfulness} = \frac{| \{ s \in S(A) : s \text{ is supported by } C \} |}{| S(A) |} $$

### 2. Answer Relevancy
Measures how directly the answer addresses the question, penalizing incomplete or redundant answers.
Let $q$ be the original question and $a$ be the generated answer. We generate $N$ potential questions $q'_i$ from $a$.

$$ \text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos\_sim(E(q), E(q'_i)) $$

Where $E(\cdot)$ is an embedding function and $\cos\_sim$ is cosine similarity.

### 3. Context Precision
Measures whether the most relevant documents are ranked highest in the retrieved context. It is a variation of Precision@K.

$$ \text{Context Precision@K} = \frac{\sum_{k=1}^{K} (Precision@k \times v_k)}{\text{Total number of relevant items in top K}} $$

Where $v_k \in \{0, 1\}$ indicates relevance of the item at rank $k$.

### 4. Context Recall
Measures if all necessary information to answer the question was retrieved.
Let $GT$ be the set of ground truth sentences/claims, and $C$ be the retrieved context.

$$ \text{Context Recall} = \frac{| \{ g \in GT : g \text{ can be attributed to } C \} |}{| GT |} $$

### End-to-End F1 Score
The harmonic mean of precision and recall for the entire pipeline.

$$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

## Query Classification Entropy
To decide if a classifier is confident in routing a query, we measure the Shannon Entropy of its prediction probabilities. Let $C$ be the set of query classes (e.g., simple, complex).

$$ H(q) = - \sum_{c \in C} P(c|q) \log_2 P(c|q) $$

High entropy means uncertainty, often triggering a fallback to a safer, more expensive retrieval strategy.

## Cache Hit Rate Optimization
The efficiency of the caching layer is simply:

$$ \text{Hit Rate} = \frac{\text{Queries served from cache}}{\text{Total queries received}} $$

## Index Staleness Metric
Staleness represents how "out of sync" the index is compared to the source of truth.

$$ \text{Staleness}(t) = \frac{\text{Documents changed since last index update}(t)}{\text{Total documents currently indexed}} $$

If $\text{Staleness}(t) > \tau$ (some threshold), an index rebuild or incremental update is triggered.

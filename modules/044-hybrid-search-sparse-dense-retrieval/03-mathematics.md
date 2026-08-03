# Mathematics of Hybrid Search

## 1. Sparse Score (BM25)

The BM25 score of a document $D$ given a query $Q$ (containing keywords $q_1, \dots, q_n$) is:

$$ \text{score}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)} $$

Where:
- $f(q_i, D)$ is the term frequency of $q_i$ in document $D$.
- $|D|$ is the length of document $D$ in words.
- $\text{avgdl}$ is the average document length in the text collection.
- $k_1$ and $b$ are free parameters, usually chosen as $k_1 \in [1.2, 2.0]$ and $b = 0.75$.
- $\text{IDF}(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)$

## 2. Dense Score (Cosine Similarity)

Given a query vector $\mathbf{q}$ and a document vector $\mathbf{d}$:

$$ \text{score}_{dense}(\mathbf{d}, \mathbf{q}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|} $$

## 3. Alpha-Blending Score Combination

$$ S_{hybrid} = \alpha S_{dense} + (1-\alpha) S_{sparse} $$

*Note: Since BM25 is unbounded and cosine similarity is bounded $[-1, 1]$, scores must be normalized (e.g., Min-Max normalization) before blending.*

## 4. Reciprocal Rank Fusion (RRF)

RRF ignores raw scores completely and relies only on the ranks $r_m(d)$ of a document $d$ in the retrieved lists $M$:

$$ \text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)} $$

Where:
- $k$ is a smoothing constant, classically set to $60$.
- $r_m(d)$ is the 1-based rank of the document in search method $m$.

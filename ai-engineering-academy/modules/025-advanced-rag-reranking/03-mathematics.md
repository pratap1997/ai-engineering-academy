# Module 025: Mathematics — Cross-Attention & Mean Average Precision (MAP)

## 1. Bi-Encoder Dot Product Similarity

Given query vector $\mathbf{e}_q = f_\theta(q) \in \mathbb{R}^d$ and document vector $\mathbf{e}_d = f_\theta(d) \in \mathbb{R}^d$:

$$\text{Score}_{\text{Bi}}(q, d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\|_2 \|\mathbf{e}_d\|_2}$$

---

## 2. Cross-Encoder Full Attention Scoring

In a Cross-Encoder, query tokens $(t_1^q, \dots, t_M^q)$ and document tokens $(t_1^d, \dots, t_N^d)$ interact directly at every layer $l \in [1, L]$ via full cross-attention:

$$\mathbf{A}_{i, j}^{(l)} = \text{softmax} \left( \frac{\mathbf{Q}_i^{(l)} (\mathbf{K}_j^{(l)})^T}{\sqrt{d_k}} \right)$$

The final relevance logit $z(q, d) \in \mathbb{R}$ is computed by linear classification head on top of the $\text{[CLS]}$ embedding:

$$\text{Score}_{\text{Cross}}(q, d) = \sigma(z(q, d)) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{h}_{\text{[CLS]}} + b)}}$$

---

## 3. Retrieval Metrics: MAP@K and MRR

Given ground-truth binary relevance labels $y_i \in \{0, 1\}$ for retrieved ranked documents at positions $i \in [1, K]$:

### Precision@K:
$$\text{P}@K = \frac{\sum_{i=1}^K y_i}{K}$$

### Mean Average Precision (MAP@K):
$$\text{AP}@K = \frac{\sum_{i=1}^K (\text{P}@i \cdot y_i)}{\sum_{i=1}^K y_i}$$

$$\text{MAP}@K = \frac{1}{|Q|} \sum_{q \in Q} \text{AP}@K(q)$$

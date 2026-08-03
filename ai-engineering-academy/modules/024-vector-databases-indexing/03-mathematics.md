# Module 024: Mathematics — Vector Metrics & HNSW Probabilistic Layers

## 1. Distance Metrics

Given query vector $\mathbf{q} \in \mathbb{R}^d$ and database vector $\mathbf{x}_i \in \mathbb{R}^d$:

### L2 Euclidean Distance:
$$D_{\text{L2}}(\mathbf{q}, \mathbf{x}_i) = \|\mathbf{q} - \mathbf{x}_i\|_2 = \sqrt{\sum_{k=1}^d (q_k - x_{i,k})^2}$$

### Cosine Similarity & Distance:
$$\text{Sim}_{\text{Cos}}(\mathbf{q}, \mathbf{x}_i) = \frac{\mathbf{q} \cdot \mathbf{x}_i}{\|\mathbf{q}\|_2 \|\mathbf{x}_i\|_2}$$

$$D_{\text{Cos}}(\mathbf{q}, \mathbf{x}_i) = 1 - \text{Sim}_{\text{Cos}}(\mathbf{q}, \mathbf{x}_i)$$

If vectors are $L_2$-normalized ($\|\mathbf{q}\| = \|\mathbf{x}\| = 1$):
$$D_{\text{L2}}^2 = 2 (1 - \mathbf{q} \cdot \mathbf{x}_i) = 2 \cdot D_{\text{Cos}}$$

---

## 2. IVF Index K-Means Voronoi Partitioning

Given dataset $\mathbf{X} \in \mathbb{R}^{N \times d}$, partition $\mathbf{X}$ into $K$ centroids $\{\mathbf{c}_1, \dots, \mathbf{c}_K\}$ minimizing variance:

$$\arg\min_{\mathbf{C}} \sum_{k=1}^K \sum_{\mathbf{x} \in S_k} \|\mathbf{x} - \mathbf{c}_k\|^2$$

Search step:
1. Find top $n_\text{probe}$ nearest centroids to query $\mathbf{q}$:
   $$\mathcal{C}_{\text{top}} = \text{TopK}_{n_\text{probe}} \left( -\|\mathbf{q} - \mathbf{c}_k\|^2 \right)$$
2. Search ONLY vectors in inverted lists of centroids $k \in \mathcal{C}_{\text{top}}$.

---

## 3. HNSW Layer Probability Assignment

In HNSW graph construction, maximum layer level $l$ for an inserted node is sampled exponentially:

$$l = \lfloor -\ln(\text{uniform}(0, 1)) \cdot m_L \rfloor$$

Where $m_L = \frac{1}{\ln(M)}$.
- Layer 0 contains ALL $N$ nodes.
- Probability of a node reaching Layer $l$ decreases exponentially: $P(\text{level} \ge l) = e^{-l / m_L} = M^{-l}$.
- Search navigates greedily down layers from top layer $L_\max$ to Layer 0.

# 03 - Mathematics

## Graph Formalisms

- **Adjacency Matrix $A$:** $A_{ij}$ is the weight of the edge from node $i$ to node $j$.
- **Degree Matrix $D$:** Diagonal matrix where $D_{ii} = \sum_j A_{ij}$.
- **Graph Laplacian $L$:** $L = D - A$.

## PageRank
$p = d M p + (1-d) v$
Where $d$ is the damping factor, $M$ is the transition matrix, and $v$ is the personalization vector (for Personalized PageRank).

## Community Detection (Leiden/Louvain)
Modularity $Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$
Where $m$ is the total edge weight, $k_i$ is the degree of node $i$, and $\delta$ is the Kronecker delta.

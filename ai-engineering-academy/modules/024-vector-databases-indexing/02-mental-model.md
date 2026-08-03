# Module 024: Mental Model — Highways & Voronoi Neighborhoods

## 1. The Highway Network Analogy (HNSW)

Imagine trying to drive to a specific house address in a foreign city:

- **Flat Index**: Driving down every single street in the entire city, checking every house one by one ($O(N)$ brute-force).
- **IVF Index**: Looking at a postal code map, identifying which neighborhood (Voronoi cell) the house is in, and driving straight to that neighborhood.
- **HNSW Index**: Taking the high-speed Interstate Highway (Layer 2, sparse long-distance connections), exiting onto the arterial boulevard (Layer 1), and then turning onto the local residential street (Layer 0, dense local connections) to arrive at the target house in **$\log(N)$ navigation steps**!

```
HNSW Multi-Layer Skip Graph:
Layer 2 (Highways):     Node 1 ───────────────────────────────> Node 850,000
                            │                                         │
Layer 1 (Boulevards):   Node 1 ──────────> Node 420,000 ──────────> Node 850,000
                            │                  │                      │
Layer 0 (Local Streets):Node 1 ─> Node 2 ─> Node 3 ... ───────────> Node 1,000,000
```

---

## 2. Voronoi Cells (IVF-Flat)

In IVF-Flat:
1. $N$ vectors are clustered into $K$ centroid points using K-Means.
2. The space is partitioned into $K$ **Voronoi cells**. Every vector belongs to the cell of its nearest centroid.
3. During search, the query vector is assigned to its $n_\text{probe}$ nearest centroids. Only vectors within those $n_\text{probe}$ cells are evaluated!

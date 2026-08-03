# Module 014: Mental Model — Clock Hands, Distance Penalties & Relative Bucketing

## 1. RoPE: Rotating Vectors on a Clock Face

Imagine each 2D pair of dimensions in a Query vector $\mathbf{q}$ or Key vector $\mathbf{k}$ as the hands of a clock.

- Token at position $m=0$: Clock hand points at $0^\circ$.
- Token at position $m=1$: Clock hand rotated by angle $\theta$.
- Token at position $m=2$: Clock hand rotated by angle $2\theta$.

When we compute the dot product between $\mathbf{q}_m$ (rotated by $m\theta$) and $\mathbf{k}_n$ (rotated by $n\theta$):

$$\mathbf{R}_{\Theta, m} \mathbf{q}_m \cdot \mathbf{R}_{\Theta, n} \mathbf{k}_n = \mathbf{q}_m^T \mathbf{R}_{\Theta, n - m} \mathbf{k}_n$$

The inner product depends **only on the relative angle difference $(m - n)\theta$**!
No matter where in the sequence two tokens appear, their relative similarity depends solely on their distance $|m - n|$.

---

## 2. ALiBi: The "Distance Penalty" Toll

ALiBi takes an even simpler approach: **don't add positional encodings to token vectors at all!**

Instead, penalize distant tokens directly in the attention matrix:

```
Raw Attention Scores S = QK^T / sqrt(d_k)

ALiBi Distance Matrix (T=4):
[ 0  -1  -2  -3 ]
[-1   0  -1  -2 ]
[-2  -1   0  -1 ]
[-3  -2  -1   0 ]

Head Slope m_h (e.g. 1/2, 1/4, 1/8):
S_modified = S - m_h * Distance Matrix
```

Tokens that are further apart pay a higher penalty in attention logits, forcing heads to naturally focus on local or semi-local context while enabling arbitrary length extrapolation.

---

## 3. T5 Relative Position Bias: Logarithmic Distance Bucketing

Relative position bucketing groups distances $(m - n)$ into discrete buckets:
- Distances $0, 1, 2, 3, 4$ get exact buckets $0, 1, 2, 3, 4$.
- Distances $> 4$ are bucketed logarithmically (e.g., distances 10–15 fall into the same bucket).

This reflects the intuition that the exact difference between position 1 and 2 matters a lot, but the difference between position 500 and 505 matters very little.

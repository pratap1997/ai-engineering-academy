# Module 010: Mathematics — Scaled Dot-Product & Multi-Head Attention

## 1. Scaled Dot-Product Attention

Given input matrices:
- $\mathbf{Q} \in \mathbb{R}^{N \times T_q \times d_k}$ — Queries
- $\mathbf{K} \in \mathbb{R}^{N \times T_k \times d_k}$ — Keys  
- $\mathbf{V} \in \mathbb{R}^{N \times T_k \times d_v}$ — Values

**Step 1**: Compute raw attention scores (similarity matrix):
$$\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} \in \mathbb{R}^{N \times T_q \times T_k}$$

**Step 2**: Apply optional causal mask (for autoregressive/decoder attention):
$$\mathbf{S}_{\text{masked}} = \mathbf{S} - \infty \cdot \mathbf{M}_\text{causal}$$

**Step 3**: Normalize with softmax over the key dimension $T_k$:
$$\mathbf{A} = \text{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times T_q \times T_k}$$

**Step 4**: Weighted sum of Values:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathbf{A} \cdot \mathbf{V} \in \mathbb{R}^{N \times T_q \times d_v}$$

---

## 2. Proof: $\sqrt{d_k}$ Prevents Softmax Saturation

For random vectors $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ with $\mu=0, \sigma^2=1$:

$$\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i \implies \mathbb{E}[\mathbf{q} \cdot \mathbf{k}] = 0, \quad \text{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k$$

Without scaling, standard deviation $= \sqrt{d_k}$, causing large scores for large $d_k$.
After dividing by $\sqrt{d_k}$:
$$\text{Var}\!\left(\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1 \quad \checkmark$$

The scores remain unit variance regardless of $d_k$, keeping the softmax in its **high-gradient regime**.

---

## 3. Multi-Head Attention

For $H$ heads each of dimension $d_\text{head} = d_\text{model} / H$:

$$\text{head}_h = \text{Attention}(\mathbf{Q}\mathbf{W}_Q^h, \mathbf{K}\mathbf{W}_K^h, \mathbf{V}\mathbf{W}_V^h)$$

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_H)\mathbf{W}_O$$

Where:
- $\mathbf{W}_Q^h, \mathbf{W}_K^h \in \mathbb{R}^{d_\text{model} \times d_\text{head}}$
- $\mathbf{W}_V^h \in \mathbb{R}^{d_\text{model} \times d_\text{head}}$
- $\mathbf{W}_O \in \mathbb{R}^{H \cdot d_\text{head} \times d_\text{model}}$

**Parameter count**: $4 \times d_\text{model}^2$ (same as a single dense head of dimension $d_\text{model}$!)

---

## 4. Sinusoidal Positional Encoding

Since attention is order-invariant (set operation), positions must be injected:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)$$

The model can learn relative positions because:
$$PE_{pos+k} = f(PE_{pos}) \text{ — a linear transformation}$$

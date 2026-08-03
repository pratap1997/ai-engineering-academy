# Module 020: Mathematics — Latent Compression & Decoupled RoPE

## 1. Low-Rank Key-Value Joint Compression

Given input token sequence $\mathbf{X} \in \mathbb{R}^{T \times d_\text{model}}$, Key-Value joint compression projects $\mathbf{X}$ into latent vector $\mathbf{c}_{KV}$:

$$\mathbf{c}_{KV} = \mathbf{X} \mathbf{W}_{DKV} \in \mathbb{R}^{T \times d_c}$$

Where $\mathbf{W}_{DKV} \in \mathbb{R}^{d_\text{model} \times d_c}$ is the down-projection matrix ($d_c \ll n_h \cdot d_h$).

During training, uncompressed Key and Value heads are reconstructed via up-projections:

$$\mathbf{K}^C = \mathbf{c}_{KV} \mathbf{W}_{UK} \in \mathbb{R}^{T \times (n_h \cdot d_h^C)}$$
$$\mathbf{V}^C = \mathbf{c}_{KV} \mathbf{W}_{UV} \in \mathbb{R}^{T \times (n_h \cdot d_h^V)}$$

---

## 2. Decoupled Rotary Positional Embeddings (RoPE)

Because RoPE is position-sensitive ($\mathbf{R}_{\Theta, m}$ varies per token position $m$), RoPE cannot be directly absorbed into static projection matrices.

DeepSeek splits Key and Query representations into two components:
1. **Content Component**: $\mathbf{K}^C$ and $\mathbf{Q}^C$ (compressed & absorbable).
2. **RoPE Component**: $\mathbf{k}^R$ and $\mathbf{q}^R$ (decoupled positional features of dimension $d_R$).

$$\mathbf{k}^R = \text{RoPE}(\mathbf{X} \mathbf{W}_{KR}) \in \mathbb{R}^{T \times d_R}$$
$$\mathbf{q}^R = \text{RoPE}(\mathbf{c}_Q \mathbf{W}_{QR}) \in \mathbb{R}^{T \times d_R}$$

### Full Combined Attention Score:
$$\mathbf{S}_{i, j} = \frac{1}{\sqrt{d_h^C + d_R}} \left( \mathbf{q}_{i, h}^C (\mathbf{k}_{j, h}^C)^T + \mathbf{q}_{i, h}^R (\mathbf{k}_{j}^R)^T \right)$$

---

## 3. Matrix Absorption Transformation for Zero-Overhead Inference

Let $\mathbf{W}_{absorbed} = \mathbf{W}_{UQ} \mathbf{W}_{UK}^T \in \mathbb{R}^{(n_h \cdot d_c) \times d_c}$.

Inference score calculation simplifies to:

$$\mathbf{q}^C (\mathbf{k}^C)^T = (\mathbf{c}_Q \mathbf{W}_{UQ}) (\mathbf{c}_{KV} \mathbf{W}_{UK})^T = \mathbf{c}_Q \mathbf{W}_{absorbed} \mathbf{c}_{KV}^T$$

**Memory Saved**:
Instead of caching $n_h \cdot d_h + n_h \cdot d_h$ numbers per token, the GPU caches ONLY $\mathbf{c}_{KV} \in \mathbb{R}^{d_c}$ and $\mathbf{k}^R \in \mathbb{R}^{d_R}$.

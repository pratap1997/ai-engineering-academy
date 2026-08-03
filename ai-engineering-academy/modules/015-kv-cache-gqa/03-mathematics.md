# Module 015: Mathematics — KV Cache & GQA Equations

## 1. KV Cache State Equations

For step $t$, incoming single token representation $\mathbf{x}_t \in \mathbb{R}^{1 \times d_\text{model}}$:

$$\mathbf{q}_t = \mathbf{x}_t \mathbf{W}_Q \in \mathbb{R}^{1 \times H_Q \times d_k}$$
$$\mathbf{k}_t = \mathbf{x}_t \mathbf{W}_K \in \mathbb{R}^{1 \times H_{KV} \times d_k}$$
$$\mathbf{v}_t = \mathbf{x}_t \mathbf{W}_V \in \mathbb{R}^{1 \times H_{KV} \times d_k}$$

**KV Cache Update**:
$$\mathbf{K}_{1:t} = \text{Concat}(\mathbf{K}_{1:t-1}, \mathbf{k}_t) \in \mathbb{R}^{t \times H_{KV} \times d_k}$$
$$\mathbf{V}_{1:t} = \text{Concat}(\mathbf{V}_{1:t-1}, \mathbf{v}_t) \in \mathbb{R}^{t \times H_{KV} \times d_k}$$

---

## 2. GQA Key/Value Expansion

Let $G = H_Q / H_{KV}$ be the repetition factor (number of Query heads per KV head).

Repeat each KV head $G$ times along the head dimension:

$$\mathbf{K}_{\text{expanded}} = \text{Repeat}(\mathbf{K}_{1:t}, \text{repeats}=G) \in \mathbb{R}^{t \times H_Q \times d_k}$$
$$\mathbf{V}_{\text{expanded}} = \text{Repeat}(\mathbf{V}_{1:t}, \text{repeats}=G) \in \mathbb{R}^{t \times H_Q \times d_k}$$

Attention score computation for the single query token $\mathbf{q}_t$:

$$\mathbf{S}_t = \frac{\mathbf{q}_t \mathbf{K}_{\text{expanded}}^T}{\sqrt{d_k}} \in \mathbb{R}^{1 \times H_Q \times t}$$
$$\mathbf{A}_t = \text{softmax}(\mathbf{S}_t, \text{axis}=-1) \in \mathbb{R}^{1 \times H_Q \times t}$$
$$\mathbf{O}_t = \mathbf{A}_t \mathbf{V}_{\text{expanded}} \in \mathbb{R}^{1 \times H_Q \times d_k}$$

---

## 3. KV Cache VRAM Memory Footprint Formula

For a model with $L$ layers, $H_{KV}$ key-value heads, head dimension $d_k$, sequence length $T$, batch size $N$, and precision $P$ bytes per element:

$$\text{VRAM}_{\text{KVCache}} = 2 \times L \times N \times T \times H_{KV} \times d_k \times P \text{ bytes}$$

### Example Calculation (LLaMA 3 70B):
- $L = 80$ layers
- $H_Q = 64$ Query heads, $H_{KV} = 8$ KV heads ($G=8$, GQA)
- $d_k = 128$
- $T = 8,192$ tokens, $N = 1$ batch, FP16 ($P=2$ bytes)

$$\text{MHA VRAM} = 2 \times 80 \times 1 \times 8192 \times 64 \times 128 \times 2 = 21.47 \text{ GB}$$
$$\text{GQA VRAM} = 2 \times 80 \times 1 \times 8192 \times 8 \times 128 \times 2 = \mathbf{2.68 \text{ GB}} \quad (8\times \text{ savings!})$$

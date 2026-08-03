# Module 017: Mathematics — Gating Router & Load Balancing Formulations

## 1. Top-k Gating Router Mathematics

Given token representation $\mathbf{x} \in \mathbb{R}^{d_\text{model}}$ and router weight matrix $\mathbf{W}_r \in \mathbb{R}^{d_\text{model} \times E}$ (where $E$ is total experts):

$$H(\mathbf{x})_i = (\mathbf{x} \mathbf{W}_r)_i + \epsilon \cdot \text{Softplus}((\mathbf{x} \mathbf{W}_\text{noise})_i)$$

Where $\epsilon \sim \mathcal{N}(0, 1)$ adds tunable exploration noise during training.

### Top-$k$ Selection & Softmax
Select top $k$ indices $\mathcal{T} = \text{TopK}(H(\mathbf{x}), k)$:

$$g_i(\mathbf{x}) = \begin{cases}
\frac{\exp(H(\mathbf{x})_i)}{\sum_{j \in \mathcal{T}} \exp(H(\mathbf{x})_j)} & \text{if } i \in \mathcal{T} \\
0 & \text{otherwise}
\end{cases}$$

---

## 2. MoE Layer Output Formulation

The output of the MoE layer is a weighted sum over the top $k$ activated expert FFNs:

$$\mathbf{y} = \sum_{i \in \mathcal{T}} g_i(\mathbf{x}) \cdot \text{Expert}_i(\mathbf{x})$$

Where each $\text{Expert}_i(\mathbf{x}) = \text{GELU}(\mathbf{x} \mathbf{W}_{1,i} + \mathbf{b}_{1,i}) \mathbf{W}_{2,i} + \mathbf{b}_{2,i}$.

---

## 3. Auxiliary Load Balancing Loss (Switch Transformer Formulation)

For a batch of $N$ tokens and $E$ experts:

Let $f_i$ be the fraction of tokens routed to expert $i$:
$$f_i = \frac{1}{N} \sum_{n=1}^N \mathbb{I}(\text{Expert } i \in \text{TopK}(\mathbf{x}_n))$$

Let $P_i$ be the average gating probability assigned to expert $i$ across all tokens:
$$P_i = \frac{1}{N} \sum_{n=1}^N \text{Softmax}(\mathbf{x}_n \mathbf{W}_r)_i$$

The auxiliary loss is:

$$\mathcal{L}_\text{aux} = \alpha \cdot E \sum_{i=1}^E f_i \cdot P_i$$

Where $\alpha$ is a scaling hyperparameter (typically $\alpha = 0.01$).

### Minimization Invariant:
By Cauchy-Schwarz, $\sum f_i P_i$ reaches its minimum when $f_i = \frac{1}{E}$ and $P_i = \frac{1}{E}$ for all $i$ (perfect uniform balance across all experts).

# 03 - Mathematics: Formalizing Agent Memory

## Working Memory Formalization

Working memory (the context window) can be formalized as a FIFO (First-In, First-Out) queue with a strict capacity $K$.

Let $W_t = [m_1, m_2, \dots, m_k]$ be the state of working memory at time step $t$, where $k \leq K$. 
When a new observation $o_{t+1}$ arrives:
- If $k < K$, $W_{t+1} = W_t \oplus [o_{t+1}]$
- If $k = K$, $W_{t+1} = [m_2, m_3, \dots, m_K] \oplus [o_{t+1}]$

## Memory Importance Score

Not all memories are equally valuable. The retrieval system relies on an Importance Score $I(m)$, often calculated as a linear combination of Recency, Frequency, and Relevance to the current query $q$:

$$ I(m, q) = \alpha \cdot \text{recency}(m) + \beta \cdot \text{frequency}(m) + \gamma \cdot \text{relevance}(m, q) $$

Where:
- $\alpha, \beta, \gamma$ are tunable hyperparameters.
- $\text{recency}(m) = e^{-\lambda (t_{current} - t_m)}$ (exponential decay over time)
- $\text{frequency}(m) = \log(1 + \text{access\_count}(m))$ (diminishing returns on access count)

## Episodic Memory Retrieval

When an agent needs to retrieve context for a query $q$, it searches the episodic memory store $M$. Retrieval is typically framed as finding the memory $m$ that maximizes cosine similarity in a high-dimensional embedding space.

$$ m^* = \text{argmax}_{m \in M} \text{sim}(E(q), E(m)) $$

Where $E(\cdot)$ is the embedding function and $\text{sim}(\mathbf{u}, \mathbf{v})$ is Cosine Similarity:

$$ \text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^n u_i v_i}{\sqrt{\sum_{i=1}^n u_i^2} \sqrt{\sum_{i=1}^n v_i^2}} $$

## Ebbinghaus Forgetting Curve

The biological forgetting curve dictates how memory retention decays over time. It can be modeled as:

$$ R = e^{-\frac{t}{S}} $$

Where:
- $R$ is memory retention (probability of recall).
- $t$ is the elapsed time since the memory was formed or last accessed.
- $S$ is the relative strength of the memory (importance).

## Memory Consolidation (Spaced Repetition)

To prevent important memories from fading, memory systems use a mechanism analogous to spaced repetition. Every time a memory is accessed, its strength $S$ is updated, flattening the forgetting curve:

$$ S_n = S_{n-1} \times k $$

Where $k > 1$. Thus, frequently accessed memories decay much more slowly.

## Hopfield Network Energy Function

Hopfield networks are a classic form of artificial associative memory. They are recurrent neural networks where all connections are symmetric. The network seeks to minimize an energy function, allowing it to "settle" into a previously memorized state when given a noisy input.

The energy $E$ of the network state $\mathbf{s} \in \{-1, 1\}^N$ is:

$$ E = -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N w_{ij} s_i s_j + \sum_{i=1}^N \theta_i s_i $$

Where:
- $w_{ij}$ is the connection weight between neuron $i$ and neuron $j$.
- $s_i$ is the state of neuron $i$.
- $\theta_i$ is the threshold of neuron $i$ (often set to 0).

## KV-Cache as Working Memory

In Transformer architectures, the Key-Value (KV) cache functions as a low-level working memory. It stores the key and value representations of previous tokens to avoid recomputing them.

For a sequence of tokens, the attention output at step $t$ uses the cached keys and values up to $t$:

$$ \text{Attention}(\mathbf{q}_t, \mathbf{K}_{1:t}, \mathbf{V}_{1:t}) = \text{softmax}\left( \frac{\mathbf{q}_t \mathbf{K}_{1:t}^T}{\sqrt{d_k}} \right) \mathbf{V}_{1:t} $$

Where $\mathbf{K}_{1:t}$ and $\mathbf{V}_{1:t}$ represent the "working memory" of the attention mechanism, bounded by the context window limit.

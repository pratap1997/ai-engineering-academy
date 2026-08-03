# Module 013: Mathematics — Tokenization Algorithms & Scoring

## 1. BPE Merge Rule Mathematics

Let $C$ be a corpus represented as sequence of symbol lists.
Count adjacent pairs $(s_i, s_{i+1})$ across all words weighted by word frequency $f(w)$:

$$\text{freq}(A, B) = \sum_{w \in C} f(w) \cdot \text{count}_{w}(A, B)$$

Select best pair $(A^*, B^*)$:
$$(A^*, B^*) = \arg\max_{(A, B)} \text{freq}(A, B)$$

Add new symbol $AB$ to vocabulary $V$ and replace all adjacent $(A, B)$ occurrences in corpus.

---

## 2. WordPiece Score Formula

For candidate subwords $A$ and $B$:

$$\text{Score}(A, B) = \frac{P(A, B)}{P(A) P(B)} = \frac{\text{count}(A, B) \cdot N}{\text{count}(A) \cdot \text{count}(B)}$$

Where $N$ is total token count. Higher score means $A$ and $B$ co-occur much more often than random chance.

---

## 3. Unigram Model & Viterbi Segmentation

For word $w$, let $\mathbf{x} = (x_1, x_2, \dots, x_k)$ be a valid subword segmentation ($w = x_1 x_2 \dots x_k$).

Probability of segmentation $\mathbf{x}$:
$$P(\mathbf{x}) = \prod_{i=1}^k P(x_i)$$

The optimal segmentation $\mathbf{x}^*$ maximizes probability (or minimizes negative log-likelihood):

$$\mathbf{x}^* = \arg\min_{\mathbf{x} \in \text{Seg}(w)} \sum_{i=1}^k -\log P(x_i)$$

Solved efficiently using the **Viterbi algorithm** (Dynamic Programming):
Let $dp[j]$ be the minimum cost to segment prefix $w[0:j]$:

$$dp[j] = \min_{0 \le i < j, w[i:j] \in V} \left( dp[i] - \log P(w[i:j]) \right)$$

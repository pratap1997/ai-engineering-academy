# Module 012: Mathematics — MLM Loss, CLM Loss & NSP

## 1. Masked Language Model (MLM) Loss — BERT

Given token sequence $\mathbf{x} = [x_1, \dots, x_T]$, let $\mathcal{M}$ be the set of masked positions.

For each masked position $i \in \mathcal{M}$, the model predicts a distribution over the vocabulary $V$:

$$p(x_i | \mathbf{x}_{\setminus \mathcal{M}}) = \text{softmax}(\mathbf{h}_i \mathbf{W}_\text{vocab}^T)$$

The MLM loss is cross-entropy averaged over **masked positions only**:

$$\mathcal{L}_\text{MLM} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log p(x_i | \mathbf{x}_{\setminus \mathcal{M}})$$

---

## 2. Causal Language Model (CLM) Loss — GPT

Given token sequence $\mathbf{x} = [x_1, \dots, x_T]$, GPT predicts each $x_{t+1}$ given $x_{\leq t}$:

$$p(x_{t+1} | x_1, \dots, x_t) = \text{softmax}(\mathbf{h}_t \mathbf{W}_\text{vocab}^T)$$

The CLM loss averages cross-entropy over **all positions** (shifted by 1):

$$\mathcal{L}_\text{CLM} = -\frac{1}{T-1} \sum_{t=1}^{T-1} \log p(x_{t+1} | x_{\leq t})$$

In practice: given logits $\mathbf{L} \in \mathbb{R}^{T \times V}$, targets are $[x_2, x_3, \dots, x_T]$.

---

## 3. Next Sentence Prediction (NSP) — BERT Auxiliary Task

BERT's second pre-training task: given two sentences $A, B$, predict if $B$ follows $A$ in the corpus.

$$p(\text{IsNext}) = \text{sigmoid}(\mathbf{W}_\text{NSP} \cdot \mathbf{h}_{[\text{CLS}]})$$

$$\mathcal{L}_\text{NSP} = -[y \log p + (1-y) \log(1-p)]$$

**Note**: Later work (RoBERTa) found NSP unhelpful and removed it. MLM alone is sufficient.

---

## 4. BERT Embeddings

$$\mathbf{E}(t) = \text{TokenEmb}(x_t) + \text{SegmentEmb}(s_t) + \text{PositionEmb}(t)$$

Where:
- **TokenEmb**: Learned $|V| \times d_\text{model}$ embedding matrix
- **SegmentEmb**: Learned 2-class embedding (sentence A=0, sentence B=1)
- **PositionEmb**: Learned (not sinusoidal) $T_\text{max} \times d_\text{model}$ matrix

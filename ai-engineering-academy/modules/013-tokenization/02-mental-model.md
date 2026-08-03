# Module 013: Mental Model — Merge Rules, Subword Trees & Dynamic Programming

## 1. Byte-Pair Encoding (BPE): Bottom-Up Merging

BPE starts from individual characters and iteratively merges the most frequent adjacent pair:

```
Corpus: "low lower newest widest"

Initial Vocab: ['l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'i', 'd']

Step 1: Most frequent pair ('e', 'r') -> Create 'er'
Step 2: Most frequent pair ('e', 's') -> Create 'es'
Step 3: Most frequent pair ('e', 'st') -> Create 'est'
Step 4: Most frequent pair ('l', 'o') -> Create 'lo'
...
Final Vocab: ['l', 'o', 'w', 'er', 'lo', 'low', 'est', 'newest', ...]
```

During **encoding**, BPE applies learned merge rules in exact priority order.

---

## 2. WordPiece: Likelihood-Driven Merging

Instead of picking the most frequent pair $(A, B)$, WordPiece picks the pair that maximizes the likelihood of the language model if merged:

$$\text{Score}(A, B) = \frac{\text{count}(A, B)}{\text{count}(A) \times \text{count}(B)}$$

This prioritizes pairs that appear together **far more often than expected by chance**, avoiding merging very common individual letters that just happen to appear frequently near each other.

BERT uses `##` to denote continuation subwords:
- `"unaffordable"` → `["un", "##afford", "##able"]`

---

## 3. Unigram: Top-Down Pruning with Viterbi Parsing

Unigram works in reverse:
1. Start with a huge initial vocabulary (all characters + frequent substrings).
2. Assign each token $x_i$ a probability $p(x_i)$.
3. Use **Viterbi Dynamic Programming** to find the highest-probability segmentation for any word.
4. Compute loss contribution of each subword and prune the bottom 10–20% least useful tokens.
5. Repeat until vocabulary reaches target size.

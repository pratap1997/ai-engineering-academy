# Module 013: Assessment & Readiness Check

## 1. Formative Questions

### Q1: Why do modern LLMs use Byte-level BPE instead of Word-level BPE?
**Answer**: Byte-level BPE operates on raw UTF-8 bytes (base vocab of 256). This eliminates Out-of-Vocabulary `[UNK]` tokens entirely while allowing the tokenizer to seamlessly handle any language, code, emojis, or arbitrary binary data without specialized vocabulary rules.

### Q2: What is the difference between WordPiece and BPE?
**Answer**: BPE merges pairs based purely on **pair frequency** (most frequent adjacent pair first). WordPiece merges pairs based on a **likelihood ratio score**: $\text{count}(A,B) / (\text{count}(A) \times \text{count}(B))$, favoring pairs whose co-occurrence is far above random chance.

### Q3: How does Unigram tokenization differ from BPE?
**Answer**: BPE is a **bottom-up, deterministic, rule-based** iterative merge strategy. Unigram is a **top-down, probabilistic** model: it starts with a huge vocabulary and uses Viterbi dynamic programming to select subword segmentations that maximize text probability, iteratively pruning low-utility subwords.

---

## 2. Capability Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands the need for subword tokenization over word/char level |
| **Competent** | Can implement `BPETokenizer` and `WordPieceTokenizer` from scratch |
| **Master** | Can build a Byte-level BPE tokenizer with 0% OOV error rate and implement Unigram Viterbi DP segmentation |

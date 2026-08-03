# Module 013: Subword Tokenization (BPE, WordPiece & Unigram)

> "Subword tokenization bridges character-level and word-level representations. It eliminates Out-of-Vocabulary (OOV) errors while keeping vocabulary sizes manageable (30k-100k tokens), forming the mandatory entry point for all modern language models."

---

## 1. Motivation: Why Subwords?

Early NLP faced a fundamental trade-off:
- **Word-Level Tokenization**: High semantic clarity, but massive vocabulary sizes ($>1\text{M}$ words) and frequent Out-of-Vocabulary (OOV) `[UNK]` errors on novel words, typos, or morphologically rich languages.
- **Character-Level Tokenization**: Zero OOV errors, but extremely long sequence lengths and weak per-token semantic signal.

**Subword Tokenization** solves this:
- Common words remain single tokens (`"the"`, `"cat"`).
- Rare or compound words are decomposed into frequent subword chunks (`"unhappiness"` → `"un"`, `"happi"`, `"ness"`).
- Unknown strings fall back to single character bytes — **0% OOV error rate**.

---

## 2. Three Tokenization Paradigms

| Algorithm | Used In | Strategy |
|---|---|---|
| **Byte-Pair Encoding (BPE)** | GPT-2, GPT-4, LLaMA, RoBERTa | Iteratively merge most frequent adjacent symbol pair |
| **WordPiece** | BERT, DistilBERT | Iteratively merge pair that maximizes likelihood gain |
| **Unigram** | SentencePiece, T5, ALBERT | Start with large vocabulary, iteratively prune tokens that minimize entropy loss |

---

## 3. Learning Outcomes

1. **Implement `BPETokenizer`**: Train vocabulary via pair frequency merges; encode and decode text.
2. **Implement `WordPieceTokenizer`**: Understand score-based merging with prefix/suffix markers (`##`).
3. **Implement `UnigramTokenizer`**: Viterbi segmentation for optimal subword parsing.
4. **Compare Tokenization Efficiency**: Measure compression ratio (bytes per token) across algorithms and languages.

---

## 4. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → Merge trees, prefix matching, and subword segmentation
03-mathematics.md       → BPE pair frequency, WordPiece likelihood, Unigram Viterbi DP
04-implementation.py    → BPETokenizer, WordPieceTokenizer, UnigramTokenizer
05-experiments.py       → Compression ratio benchmark & out-of-vocabulary handling
06-real-applications.md → tiktoken, HuggingFace tokenizers, Byte-level BPE
07-engineering-challenge.md → Build a byte-level BPE tokenizer with fallback
08-assessment.md        → Readiness check
09-references.md        → Sennrich et al. (2016) & Kudo (2018)
```

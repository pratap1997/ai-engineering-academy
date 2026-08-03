# Module 012: Overview — BERT & GPT Pre-training Objectives

> "The single most important insight in NLP since 2018: you don't need task-specific labeled data to build powerful language representations. Pre-train on raw text at massive scale, then fine-tune with a handful of examples."

---

## 1. The Pre-training Hypothesis

Before BERT and GPT, NLP models were trained from scratch on small labeled datasets for each task (sentiment, NER, translation, QA). This was sample-inefficient and fragile.

The **pre-training hypothesis**:
- Language structure, facts, reasoning patterns, and syntax are all implicit in raw text.
- A model trained to predict masked/next tokens on billions of words will learn general representations.
- These representations transfer to downstream tasks with far less labeled data.

**Result**: BERT (2018) improved state-of-the-art on 11 NLP tasks simultaneously. GPT-3 (2020) solved tasks it had never been trained on explicitly.

---

## 2. Two Paradigms

| | **BERT** | **GPT** |
|---|---|---|
| **Pre-training task** | Masked Language Model (MLM) + NSP | Causal Language Model (CLM) |
| **Attention** | Bidirectional (sees full context) | Causal/Autoregressive (past only) |
| **Architecture** | Encoder-only | Decoder-only |
| **Best for** | Understanding (classification, NER, QA) | Generation (text, code, reasoning) |

---

## 3. Learning Outcomes

1. **Implement `MLMMasker`**: Apply the 80/10/10 masking strategy to token sequences.
2. **Implement `MLMLoss`** and **`CLMLoss`**: Masked cross-entropy for BERT; shifted cross-entropy for GPT.
3. **Implement `BERTEmbeddings`**: Token + Segment + Position embeddings combined.
4. **Explain** why the pre-train → fine-tune paradigm works from a representation learning perspective.

---

## 4. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → Fill-in-the-blank (BERT) vs. predicting the next word (GPT)
03-mathematics.md       → MLM loss, CLM loss, NSP binary cross-entropy
04-implementation.py    → MLMMasker, MLMLoss, CLMLoss, BERTEmbeddings
05-experiments.py       → MLM 80/10/10 ratio verification, loss curves
06-real-applications.md → HuggingFace BERT fine-tuning, GPT text generation
07-engineering-challenge.md → Mini MLM pre-training loop on toy vocabulary
08-assessment.md        → Readiness check
09-references.md        → Devlin et al. (2019), Radford et al. (2018/2019)
```

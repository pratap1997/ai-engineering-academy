# Module 012: Assessment

## 1. Formative Questions

### Q1: Why mask only 15% of tokens, not more?
**Answer**: Masking too many tokens would make the task trivially hard (too much context missing) and slow convergence. Masking too few wastes compute per batch. 15% was found empirically to balance learning signal density vs. context availability.

### Q2: Why does GPT use causal masking during pre-training?
**Answer**: GPT is trained to predict the next token autoregressively. If it could see future tokens during training, it would trivially copy them at inference time (where future tokens don't exist). Causal masking mirrors the inference-time constraint.

### Q3: What does BERT's [CLS] token represent?
**Answer**: `[CLS]` (classification token) is prepended to every input. Its final hidden state aggregates information from the entire sequence via bidirectional self-attention. It serves as the sentence-level representation for downstream tasks like classification and NSP.

### Q4: Why is RoBERTa better than BERT despite same architecture?
**Answer**: (1) No NSP task — MLM alone is sufficient. (2) 10x more data. (3) Larger batch sizes. (4) Longer training. (5) Dynamic masking — masks change each epoch instead of being fixed. Architecture is identical — it's all training procedure and data.

---

## 2. Capability Rubric

| Level | Criteria |
|---|---|
| **Novice** | Can explain MLM and CLM conceptually |
| **Competent** | Can implement `MLMMasker` with 80/10/10 and compute masked cross-entropy |
| **Master** | Can implement full BERT embeddings, CLM shifted loss, and explain why pre-training works from a representation learning perspective |

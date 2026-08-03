# Module 015: Mental Model — The Scratchpad Notebook & Head Grouping

## 1. The Scratchpad Notebook Analogy

Imagine taking an open-book exam where you write one word per minute:

- **Without KV Cache**: Every minute, you erase your entire notepad, re-read all 1,000 previous words from scratch, re-calculate all key notes, and then write word 1,001. By minute 2,000, you are spending 99.9% of your time re-reading past notes!
- **With KV Cache**: You keep a running notepad of key notes ($K$) and summaries ($V$). Every minute, you compute key notes *only* for the single new word, append them to the notepad, and glance at the notepad to write the next word.

---

## 2. Grouped-Query Attention (GQA) Head Mapping

In standard MHA with 8 heads, each Query head has its own private Key and Value notebook:

```
Multi-Head Attention (MHA):
Q_0 ──→ K_0, V_0
Q_1 ──→ K_1, V_1
Q_2 ──→ K_2, V_2
Q_3 ──→ K_3, V_3
Q_4 ──→ K_4, V_4
Q_5 ──→ K_5, V_5
Q_6 ──→ K_6, V_6
Q_7 ──→ K_7, V_7
(8 KV heads stored in VRAM)
```

In Grouped-Query Attention (GQA) with 2 KV groups (ratio 4:1):

```
Grouped-Query Attention (GQA):
Q_0, Q_1, Q_2, Q_3  ──→ Group 0 (K_0, V_0)
Q_4, Q_5, Q_6, Q_7  ──→ Group 1 (K_1, V_1)
(Only 2 KV heads stored in VRAM -> 4x VRAM saving!)
```

In Multi-Query Attention (MQA) with 1 KV group:

```
Multi-Query Attention (MQA):
Q_0, Q_1, Q_2, Q_3, Q_4, Q_5, Q_6, Q_7 ──→ Group 0 (K_0, V_0)
(Only 1 KV head stored in VRAM -> 8x VRAM saving!)
```

# Module 015: KV Cache & Grouped-Query Attention (GQA / MQA / MHA)

> "During autoregressive text generation, recomputing attention over past tokens at every new step is an $O(T^2)$ memory bandwidth disaster. KV Caching reduces generation step complexity from $O(T)$ to $O(1)$, while Grouped-Query Attention (GQA) reduces KV memory footprint by $4\times$--$8\times$ without losing model quality."

---

## 1. Motivation: The Memory Bandwidth Bottleneck

In LLM inference (GPT-4, LLaMA 3), token generation happens one token at a time.
Without a KV cache, generating token $t=1001$ requires re-running the entire Transformer forward pass on all 1,000 previous tokens!

1. **KV Cache**: Stores previous Key and Value states in memory so each new decoding step only computes $Q, K, V$ for the single new incoming token.
2. **Grouped-Query Attention (GQA)**: In standard Multi-Head Attention (MHA), each Query head has its own Key and Value head ($H_Q = H_{KV}$). For large models (e.g. 70B params), storing 64 KV heads in VRAM exhausts GPU memory. GQA shares 1 KV head across $G$ Query heads (e.g. 8 Query heads per 1 KV head), slashing KV cache VRAM requirements by $8\times$.

---

## 2. Attention Spectrum

$$\text{MHA } (H_{KV} = H_Q) \quad \longleftarrow \quad \text{GQA } (1 < H_{KV} < H_Q) \quad \longleftarrow \quad \text{MQA } (H_{KV} = 1)$$

- **MHA (Multi-Head Attention)**: $H_Q$ Query heads, $H_Q$ Key/Value heads. Highest capacity, largest VRAM usage.
- **GQA (Grouped-Query Attention)**: $H_Q$ Query heads, $H_{KV} = H_Q / G$ Key/Value heads. Best quality/speed balance (LLaMA 3 default).
- **MQA (Multi-Query Attention)**: $H_Q$ Query heads, 1 Key/Value head. Maximum VRAM savings, slightly lower quality.

---

## 3. Learning Outcomes

1. **Implement `KVCache`**: Dynamic append-only state container supporting batch decoding.
2. **Implement `GroupedQueryAttention`**: Unified MHA / GQA / MQA forward pass with KV head repeating.
3. **Calculate Memory Footprint**: Formulate VRAM usage per token for KV cache across precision types (FP16/FP8/INT4).
4. **Benchmark Speedup**: Prove $O(1)$ per-step latency vs $O(T)$ naive recomputation.

---

## 4. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → The scratchpad notebook analogy & head grouping
03-mathematics.md       → KV cache update equations, GQA repeat factors, memory formulas
04-implementation.py    → KVCache, GroupedQueryAttention
05-experiments.py       → KV cache vs naive recomputation latency benchmark, VRAM calculator
06-real-applications.md → LLaMA 3 GQA config, vLLM PagedAttention integration
07-engineering-challenge.md → Incremental autoregressive text generator with KV cache
08-assessment.md        → Readiness check
09-references.md        → Ainslie et al. (2023), Shazeer (2019)
```

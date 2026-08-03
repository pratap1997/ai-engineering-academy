# Module 015: Assessment & Readiness Check

## 1. Formative Questions

### Q1: Why does autoregressive decoding suffer from a memory bandwidth bottleneck?
**Answer**: At each step of text generation, only 1 new token is processed ($O(d^2)$ compute). However, to compute attention over the past sequence, all past weight matrices and KV states must be fetched from GPU VRAM ($O(T \cdot d)$ memory transfers). Modern GPUs have immense compute capability but limited memory bandwidth, causing the GPU arithmetic units to sit idle waiting for memory.

### Q2: What is the difference between MHA, MQA, and GQA?
**Answer**:
- **Multi-Head Attention (MHA)**: $H_{KV} = H_Q$. Each Query head has its own Key and Value head. Maximum capacity, highest VRAM usage.
- **Multi-Query Attention (MQA)**: $H_{KV} = 1$. All Query heads share a single Key and Value head. Maximum VRAM savings ($H_Q \times$), slight quality loss.
- **Grouped-Query Attention (GQA)**: $1 < H_{KV} < H_Q$. Query heads are grouped into clusters sharing 1 Key and Value head. Delivers MQA-like VRAM savings with MHA-level model accuracy.

### Q3: How does PagedAttention improve KV cache memory efficiency?
**Answer**: PagedAttention eliminates memory fragmentation by allocating physical memory in fixed-size blocks (pages) dynamically as the sequence grows, rather than pre-allocating large contiguous blocks for maximum potential context length.

---

## 2. Capability Rubric

| Level | Criteria |
|---|---|
| **Novice** | Can explain why KV caching is needed conceptually during text generation |
| **Competent** | Can implement `KVCache` and `GroupedQueryAttention` from scratch in Python |
| **Master** | Can write an incremental autoregressive generator, calculate exact VRAM savings across context lengths, and explain PagedAttention block allocation |

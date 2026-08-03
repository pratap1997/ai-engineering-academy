# Module 023: Assessment & Readiness Check

## 1. Formative Questions

### Q1: Why is LLM autoregressive generation memory-bandwidth bound rather than compute-bound?
**Answer**: In standard token-by-token decoding, generating each token requires reading all model weights (e.g. 140 GB for a 70B FP16 model) from GPU High Bandwidth Memory (HBM) into Tensor Cores. The GPU arithmetic intensity is tiny ($<1\%$ Tensor Core utilization), meaning time is spent waiting on memory transfer rather than doing floating-point math.

### Q2: How does Speculative Decoding achieve speedup without degrading generation quality?
**Answer**: A tiny Draft model quickly generates $K$ candidate tokens. The large Target model evaluates all $K$ candidate tokens in a single parallel forward pass. Rejection sampling with acceptance threshold $\min\left(1, \frac{p(x)}{q(x)}\right)$ guarantees that accepted tokens recover the exact Target probability distribution $p(x)$ with zero quality loss.

### Q3: What happens when a draft token is rejected at position $M < K$?
**Answer**: When draft token $x_M$ is rejected, all candidate tokens after index $M$ are discarded. The Target model samples a replacement token $x'$ from adjusted distribution $\text{relu}(p(x) - q(x))$, and the KV Caches of both models roll back to length $M+1$.

---

## 2. Capability Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands why speculative decoding bypasses the memory-bandwidth bottleneck |
| **Competent** | Can implement `RejectionSampler` with probability threshold $\min(1, p(x)/q(x))$ |
| **Master** | Can build `KVCacheRollbackSpeculativeEngine`, implement parallel draft verification, and prove exact target distribution recovery |

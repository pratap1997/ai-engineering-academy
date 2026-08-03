# Module 023: Engineering Challenge — KV Cache Rollback Speculative Engine

## 1. Challenge Task

Construct a self-contained `KVCacheRollbackSpeculativeEngine` in pure Python & NumPy that:
1. Maintains KV Caches for both Draft and Target models.
2. Rolls back the Target model's KV Cache when candidate draft tokens $x_i$ are rejected at index $M < K$.
3. Verifies that sequence generation with speculative decoding matches sequential autoregressive target generation with zero divergence.

---

## 2. Validation Criteria

1. Target model KV Cache rolls back seamlessly upon token rejection ($M < K$).
2. Output generated tokens are 100% valid under Target model logits.
3. Zero NaNs or KV Cache memory leaks.

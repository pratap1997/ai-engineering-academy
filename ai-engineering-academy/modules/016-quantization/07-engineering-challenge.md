# Module 016: Engineering Challenge — Quantized Multi-Head Attention Layer

## 1. Challenge Task

Construct a self-contained `QuantizedMultiHeadAttention` layer in pure Python & NumPy that:
1. Replaces standard $W_Q, W_K, W_V, W_O$ weight projections with `QuantizedLinear` (INT4 group-wise quantized).
2. Performs forward pass self-attention with on-the-fly weight dequantization.
3. Compares output activations with full FP32 Multi-Head Attention to verify that relative cosine similarity between quantized and FP32 outputs exceeds $0.98$.

---

## 2. Validation Criteria

1. Memory footprint of weight matrices reduced by $4\times$ (INT4 group-wise).
2. Cosine similarity between quantized MHA output and FP32 MHA output $> 0.98$.
3. Zero NaNs or numerical degradation.

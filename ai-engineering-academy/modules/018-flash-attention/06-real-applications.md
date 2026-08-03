# Module 018: Real Applications — Production FlashAttention & Hardware Execution

## 1. PyTorch Native `scaled_dot_product_attention`

Starting in PyTorch 2.0, `torch.nn.functional.scaled_dot_product_attention` automatically dispatches to optimized FlashAttention CUDA kernels:

```python
import torch
import torch.nn.functional as F

q = torch.randn(2, 8, 2048, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 2048, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 2048, 64, device="cuda", dtype=torch.float16)

# Automatically executes FlashAttention-2 CUDA kernel under PyTorch 2.0+
output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

---

## 2. FlashAttention-2 vs FlashAttention-3 (NVIDIA Hopper)

| Version | Key Technical Advance | Speedup vs Standard | Primary Hardware Target |
|---|---|---|---|
| **FlashAttention-1 (2022)** | Tiled Online Softmax + HBM IO reduction | $2\times$--$4\times$ | NVIDIA Ampere (A100) / Turing |
| **FlashAttention-2 (2023)** | Swapped loops (parallelize over $T_r$), reduced non-matmul FLOPs | $2\times$ over FA-1 ($5\times$--$7\times$ total) | NVIDIA A100 / H100 |
| **FlashAttention-3 (2024)** | Overlaps Tensor Core matmul with TMA (Tensor Memory Accelerator) & FP8 precision | $1.5\times$--$2\times$ over FA-2 | NVIDIA Hopper (H100/H200) |

---

## 3. Why FlashAttention is Essential for Long-Context LLMs

Without FlashAttention:
- LLaMA 3 8B fine-tuning at $T=128,000$ context is **physically impossible** due to VRAM overflow ($>1\text{ TB}$ VRAM needed for attention matrices).

With FlashAttention:
- $128,000$ token context fine-tuning runs comfortably on standard 80GB H100 GPU clusters!

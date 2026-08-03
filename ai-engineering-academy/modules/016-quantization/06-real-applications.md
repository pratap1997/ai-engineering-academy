# Module 016: Real Applications — GPTQ, AWQ, SmoothQuant & bitsandbytes

## 1. `bitsandbytes` (LLM.int8() & QLoRA)

`bitsandbytes` is the core library powering HuggingFace 8-bit and 4-bit loading:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Load LLaMA 3 8B in 4-bit NF4 (NormalFloat4)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # Quantize quantization scales for extra VRAM savings
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=quantization_config,
    device_map="auto"
)
```

---

## 2. GPTQ vs AWQ vs SmoothQuant

| Technique | Used For | Key Innovation | Speed / Accuracy |
|---|---|---|---|
| **GPTQ** | 4-bit Weight-Only | Second-order Taylor optimization (Hessian inverse) to adjust remaining weights after each column is quantized | Fast generation, excellent accuracy for 3B--70B models |
| **AWQ (Activation-aware Weight Quantization)** | 4-bit Weight-Only | Protects top 1% salient weights based on activation magnitude rather than weight magnitude | Faster inference kernels than GPTQ, higher accuracy on complex reasoning |
| **SmoothQuant** | 8-bit Weight + Activation (W8A8) | Migrates quantization difficulty from activations to weights using per-channel smoothing factors | True $2\times$ GEMM compute speedup on INT8 Tensor Cores |
| **bitsandbytes (NF4)** | QLoRA Fine-tuning | Information-theoretically optimal quantile distribution for zero-mean normal weights | Industry standard for low-VRAM fine-tuning |

# Module 015: Real Applications — Production KV Caching & GQA

## 1. PyTorch & HuggingFace KV Cache Interface

```python
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

inputs = tokenizer("Artificial Intelligence is", return_tensors="pt")

# Step 1: Prefill phase (passes full prompt, returns past_key_values)
outputs = model(**inputs, use_cache=True)
past_key_values = outputs.past_key_values  # Tuple of (K, V) per layer
next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

# Step 2: Autoregressive decoding phase (single token per step with cache)
for _ in range(20):
    outputs = model(input_ids=next_token_id, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values  # Updated cache
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
```

---

## 2. vLLM & PagedAttention

Traditional KV caching allocates contiguous memory blocks for the maximum possible sequence length ($T=8192$), resulting in **60--80% memory fragmentation and waste**.

**PagedAttention** (vLLM framework) applies operating system virtual memory concepts to KV caching:
- Breaks KV cache into fixed-size physical blocks (e.g. 16 tokens per block).
- Maps logical sequence tokens to non-contiguous physical memory pages dynamically.
- Eliminates memory fragmentation and enables **$2\times$--$4\times$ throughput scaling** for production serving endpoints.

---

## 3. Production Model Architecture GQA Configs

| Model | Layers ($L$) | Query Heads ($H_Q$) | KV Heads ($H_{KV}$) | GQA Ratio | VRAM Saving |
|---|---|---|---|---|---|
| **LLaMA 2 7B** | 32 | 32 | 32 (MHA) | 1:1 | 0% |
| **LLaMA 3 8B** | 32 | 32 | 8 (GQA) | 4:1 | 75% (4x) |
| **LLaMA 3 70B** | 80 | 64 | 8 (GQA) | 8:1 | 87.5% (8x) |
| **Mistral 7B** | 32 | 32 | 8 (GQA) | 4:1 | 75% (4x) |
| **Falcon 40B** | 60 | 64 | 1 (MQA) | 64:1 | 98.4% (64x) |

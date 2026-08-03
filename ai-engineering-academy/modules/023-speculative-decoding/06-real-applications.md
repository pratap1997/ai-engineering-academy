# Module 023: Real Applications — Production Speculative Decoding & Medusa

## 1. Production Speculative Decoding Engines (vLLM & TensorRT-LLM)

Both vLLM and NVIDIA TensorRT-LLM integrate speculative decoding as a core inference acceleration feature:

```python
from vllm import LLM, SamplingParams

# Instantiate 70B Target model with 1B Draft model speculative decoding
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    speculative_model="meta-llama/Meta-Llama-3-8B-Instruct",
    num_speculative_tokens=5,
    use_v2_block_manager=True
)

prompts = ["Explain quantum computing in simple terms:"]
outputs = llm.generate(prompts, SamplingParams(temperature=0.7))
# -> Achieves 2.4x latency reduction (12ms/tok vs 29ms/tok)
```

---

## 2. Draft-Free Speculative Architectures (Medusa & Eagle-2)

- **Medusa (Cai et al., 2024)**: Eliminates the separate Draft model by attaching multiple lightweight MLP "heads" directly to the top layer of the Target model. Each head predicts tokens at position $+1, +2, +3, +4$ simultaneously!
- **Eagle-2 (Li et al., 2024)**: Uses a feature-level draft head operating on top of the Target model's second-to-last hidden state, achieving up to **$3.0\times$ speedup** across diverse LLM tasks.

---

## 3. Pairing Draft and Target Models

| Target Model | Recommended Draft Model | Typical Acceptance Rate | Speedup |
|---|---|---|---|
| **LLaMA 3 70B** | LLaMA 3 8B | $75\text{--}85\%$ | $2.2\times$ |
| **DeepSeek V3 (671B)** | DeepSeek V3 Speculative Draft (7B) | $80\text{--}88\%$ | $2.5\times$ |
| **Qwen 2.5 72B** | Qwen 2.5 1.5B | $70\text{--}80\%$ | $2.0\times$ |

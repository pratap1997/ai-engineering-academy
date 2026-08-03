# Module 017: Real Applications — Production MoE Models

## 1. Mixtral 8x7B (Mistral AI)

Mixtral 8x7B is the open-source milestone that popularized sparse MoE in frontier LLMs:

```
Total Parameters:  46.7 Billion
Active Parameters: 12.9 Billion per token
Architecture:      8 Experts per layer, Top-2 Routing (k=2)
Context Window:    32,768 tokens
Performance:       Outperforms LLaMA 2 70B while executing 6x faster!
```

---

## 2. DeepSeek V2 / V3 Fine-Grained MoE Architecture

DeepSeek V3 takes MoE to extreme efficiency with **Fine-Grained Expert Segmentation**:

```
Total Experts:           256 Experts
Active Experts:          8 Active Experts per token (k=8)
Shared Expert:           1 Shared Expert always activated for general knowledge
Load Balancing:          Auxiliary-loss-free load balancing algorithm
Active/Total Ratio:      37B active / 671B total parameters
```

---

## 3. MoE Model Architecture Comparison

| Model | Total Params | Active Params | Num Experts ($E$) | Top-$k$ | Router Loss |
|---|---|---|---|---|---|
| **Switch Transformer** | 1.6 Trillion | 1.6 Billion | 2048 | Top-1 | Switch Aux Loss |
| **Mixtral 8x7B** | 46.7 Billion | 12.9 Billion | 8 | Top-2 | Load Balancing Loss |
| **Mixtral 8x22B** | 141 Billion | 39 Billion | 8 | Top-2 | Load Balancing Loss |
| **Grok-1 (xAI)** | 314 Billion | 86 Billion | 8 | Top-2 | Load Balancing Loss |
| **DeepSeek V3** | 671 Billion | 37 Billion | 256 + 1 Shared | Top-8 | Bias-based Aux-free |

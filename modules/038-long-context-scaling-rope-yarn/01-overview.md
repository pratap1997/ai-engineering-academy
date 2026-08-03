# 01 - Overview: Long-Context Scaling & Position Encodings

## Introduction
Modern LLMs are increasingly defined by their context windows. While early models (like BERT or GPT-2) supported 512 to 1024 tokens, modern models routinely support 128k to 2M tokens. Handling such massive contexts introduces severe computational and mathematical challenges.

## The Context Length Limit
Context length limits arise from two main sources:
1. **Computational Complexity**: Standard attention scales quadratically $O(N^2)$ with sequence length $N$.
2. **Positional Generalization**: Models trained on sequences of length $L$ struggle to generalize to lengths $> L$.

## Rotary Position Embedding (RoPE) and Out-of-Distribution Lengths
RoPE represents token positions as rotations in a complex plane. When sequence lengths exceed the training length, the model encounters high-frequency rotations and relative distances it has never seen, leading to catastrophic performance degradation. 

## Scaling Techniques
To extend the context window without training from scratch, researchers developed position interpolation and scaling techniques:
- **Linear RoPE Scaling**: Directly divides the position indices by a scaling factor $s$, compressing a long sequence into the original training length.
- **NTK-aware Scaling**: Modifies the base of the RoPE frequencies, spreading out the interpolation so that high frequencies (local context) are preserved while low frequencies (global context) are interpolated.
- **YaRN (Yet Another RoPE Extrapolation)**: Improves upon NTK by applying a ramp function that separates frequencies into domains (fast, slow, mid) and applies optimal interpolation to each, coupled with temperature scaling.

## Execution Optimizations
Even with mathematical extrapolation, $O(N^2)$ attention is too slow and memory-intensive:
- **Sliding Window Attention (SWA)**: Restricts attention to a local window of past tokens (e.g., 4096 tokens), reducing complexity to $O(N \times W)$.
- **Chunked Prefill**: Splits massive prompts into smaller, manageable chunks during the prefill phase, reducing peak memory usage and preventing Out-Of-Memory (OOM) errors.

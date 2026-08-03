# 07 - Engineering Challenge: Needle-In-A-Haystack Tester

## Objective
Your task is to build a "Needle-In-A-Haystack" Benchmark Tester to evaluate how different RoPE scaling algorithms affect retrieval accuracy at long context lengths.

## Requirements
1. **Context Generation**: Create a synthetic "haystack" of text consisting of random but plausible words or tokens up to 32,000 tokens long.
2. **Needle Insertion**: Insert a specific "needle" (e.g., "The secret password is 'tesseract'.") at variable depths (0%, 25%, 50%, 75%, 100%) within the haystack.
3. **Model Integration**: Use the provided `StandardRoPE`, `LinearScaledRoPE`, and `YaRNRoPE` classes to encode the positions of the haystack.
4. **Attention Retrieval**: Simulate an attention mechanism where a Query vector searches for the "needle" tokens among the Key vectors. Measure the attention scores.
5. **Evaluation Metric**: Output a heatmap or table showing the attention score (or retrieval success rate) for each RoPE method across different sequence lengths (4k, 8k, 16k, 32k) and depths.

## Constraints
- Do not use any external ML frameworks like PyTorch or TensorFlow. Stick to numpy.
- The simulation does not need to train a model; simply calculate the mathematical dot products of the RoPE-encoded query and key vectors.

No hints provided. Good luck!

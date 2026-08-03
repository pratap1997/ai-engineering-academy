# Module 015: References

1. **Ainslie, J. et al. (2023)**. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. EMNLP.
   - **Provenance**: Introduces Grouped-Query Attention (GQA) and demonstrates 8x VRAM reduction with near-zero accuracy loss. Standard in LLaMA 3, Mistral, and Qwen 2.

2. **Shazeer, N. (2019)**. *Fast Transformer Decoding: One Write-Head is All You Need*. arXiv:1911.02150.
   - **Provenance**: Introduces Multi-Query Attention (MQA), sharing 1 KV head across all Query heads.

3. **Kwon, W. et al. (2023)**. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP.
   - **Provenance**: Introduces PagedAttention and vLLM framework for non-contiguous KV cache allocation.

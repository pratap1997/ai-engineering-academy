# Module 018: References

1. **Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022)**. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS.
   - **Provenance**: Introduces the tiled online softmax FlashAttention-1 algorithm and IO-awareness framework.

2. **Dao, T. (2023)**. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. ICLR.
   - **Provenance**: Optimizes FlashAttention loop ordering to parallelize across sequence length and improve Tensor Core utilization.

3. **Milakov, M., & Gimelshein, N. (2018)**. *Online normalizer calculation for softmax*. arXiv:1805.02867.
   - **Provenance**: Mathematical foundation for Online Softmax recurrence relations.

4. **Shah, J. et al. (2024)**. *FlashAttention-3: Fast Bellman-Optimized Attention on Hopper GPUs*. arXiv:2407.08608.
   - **Provenance**: Adapts FlashAttention for NVIDIA Hopper TMA (Tensor Memory Accelerator) and FP8 precision.

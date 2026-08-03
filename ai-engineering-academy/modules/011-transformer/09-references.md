# Module 011: References

1. **Vaswani, A. et al. (2017)**. *Attention Is All You Need*. NeurIPS.
   - Original Post-LN Transformer architecture with FFN, Multi-Head Attention, and Sinusoidal PE.

2. **Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016)**. *Layer Normalization*. arXiv:1607.06450.
   - Derivation of LayerNorm as a sequence-length-independent alternative to BatchNorm.

3. **Xiong, R. et al. (2020)**. *On Layer Normalization in the Transformer Architecture*. ICML.
   - Formal proof that Pre-LN improves gradient norm stability over Post-LN; motivates modern default of norm_first=True.

4. **Hendrycks, D. & Gimpel, K. (2016)**. *Gaussian Error Linear Units (GELUs)*. arXiv:1606.08415.
   - Introduction of GELU activation; empirically outperforms ReLU in Transformer FFNs.

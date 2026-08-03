# Module 017: References

1. **Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017)**. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR.
   - **Provenance**: Foundational paper introducing Top-k Gating Router and sparse MoE layers.

2. **Fedus, W., Zoph, B., & Shazeer, N. (2022)**. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR.
   - **Provenance**: Introduces Top-1 routing and the auxiliary load balancing loss $\mathcal{L}_\text{aux} = \alpha E \sum f_i P_i$.

3. **Jiang, A. Q. et al. (2024)**. *Mixtral of Experts*. arXiv:2401.04088.
   - **Provenance**: Technical report for Mixtral 8x7B, establishing open-weights MoE standards.

4. **DeepSeek-AI (2024)**. *DeepSeek-V3 Technical Report*. arXiv:2412.19437.
   - **Provenance**: Introduces 256 fine-grained experts with auxiliary-loss-free load balancing.

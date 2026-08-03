# Module 017: Assessment & Readiness Check

## 1. Formative Questions

### Q1: Why does sparse MoE allow building 100B+ parameter models with fast inference speeds?
**Answer**: Sparse MoE decouples total parameter capacity from compute cost per token. A model can store 8 or 64 expert FFNs in VRAM (high capacity), but for any individual token, the gating router activates only top-$k$ ($k=1$ or $2$) experts. Thus, FLOPs per token equal a 13B model even though total parameters equal 47B+.

### Q2: What is Routing Collapse and how does Auxiliary Load Balancing Loss prevent it?
**Answer**: Routing collapse occurs when the router favors 1 or 2 experts early in training, sending almost all tokens to them while other experts remain uninitialized. The auxiliary loss $\mathcal{L}_\text{aux} = \alpha E \sum f_i P_i$ penalizes unequal distribution of tokens ($f_i$) and router probabilities ($P_i$), mathematically forcing uniform token distribution across all experts.

### Q3: How does DeepSeek V3's MoE design differ from Mixtral 8x7B?
**Answer**: Mixtral uses 8 large experts with Top-2 routing (2 active per token). DeepSeek V3 uses 256 fine-grained small experts with Top-8 routing, plus 1 always-active shared expert that captures universal syntax and common facts across all tokens.

---

## 2. Capability Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands the difference between dense and sparse MoE architectures |
| **Competent** | Can implement `TopKRouter` and calculate Switch auxiliary load balancing loss |
| **Masker** | Can build a complete `MoELayer`, dispatch tokens sparsely to experts, and build a full MoE Transformer block |

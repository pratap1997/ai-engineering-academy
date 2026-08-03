# Module 017: Mixture-of-Experts (MoE) & Top-k Gating

> "Dense models compute every layer's parameter for every token. Sparse Mixture-of-Experts (MoE) routes each token to a small subset of specialized expert Feed-Forward Networks (FFNs), decoupling total model parameter capacity from per-token compute cost."

---

## 1. Motivation: Sparse Computation at Scale

To increase model intelligence, scaling parameter count is key. However, running a 100B parameter dense model requires 100B FLOPs per token.

**Mixture-of-Experts (MoE)** replaces the single dense Feed-Forward Network (FFN) in a Transformer block with:
- $E$ parallel expert FFN networks ($E = 8$ or $64$).
- A **Gating Router** that selects the top $k$ experts ($k = 1$ or $2$) for each token.

**Result**: A model with **47B total parameters** (Mixtral 8x7B) only activates **13B parameters per token**, executing at the speed of a 13B model while achieving 47B-level performance!

---

## 2. Core Challenges in MoE Architectures

1. **Routing Collapse**: Without regularization, the router learns to send all tokens to 1 or 2 favorite experts, leaving other experts untrained and idle.
2. **Load Balancing Loss**: An auxiliary loss $\mathcal{L}_\text{aux}$ added during training that penalizes uneven token distribution across experts.
3. **Capacity Factor**: Expert buffer capacity caps to prevent GPU VRAM overflow when an expert receives more tokens than expected.

---

## 3. Learning Outcomes

1. **Implement `TopKRouter`**: Softmax gating with Top-$k$ selection and routing probabilities.
2. **Implement `LoadBalancingLoss`**: Auxiliary loss penalizing imbalanced token routing.
3. **Implement `MoELayer`**: Dispatch tokens to selected experts, evaluate FFNs, and aggregate weighted outputs.
4. **Compare Active vs Total Parameters**: Calculate computational efficiency of MoE vs Dense architectures.

---

## 4. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → The hospital triage nurse & specialist doctors analogy
03-mathematics.md       → Top-k routing formulas, Gumbel noise, Switch auxiliary loss
04-implementation.py    → TopKRouter, LoadBalancingLoss, ExpertFFN, MoELayer
05-experiments.py       → Load balancing loss ablation & expert utilization benchmark
06-real-applications.md → Mixtral 8x7B, DeepSeek V2/V3 MoE, Grok-1
07-engineering-challenge.md → Build a complete MoE-enhanced Transformer block
08-assessment.md        → Readiness check
09-references.md        → Shazeer et al. (2017), Fedus et al. (2022)
```

# Module 021: Assessment & Readiness Check

## 1. Formative Questions

### Q1: How does Direct Preference Optimization (DPO) eliminate the need for a separate Reward Model?
**Answer**: DPO mathematically proves that the optimal reward function $r(x, y)$ under the RLHF KL-constrained objective can be expressed directly in terms of the policy $\pi_\theta(y \mid x)$ and reference model $\pi_\text{ref}(y \mid x)$: $r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)}$. Substituting this into the Bradley-Terry preference model allows training directly on preferred/dispreferred completion pairs with binary cross-entropy loss.

### Q2: What happens if $\beta = 0$ in DPO?
**Answer**: When $\beta \to 0$, the KL penalty term disappears, and the model focuses entirely on maximizing chosen probability while ignoring baseline model drift. This causes mode collapse, catastrophic forgetting, and severe language degradation.

### Q3: Why does DPO require keeping a copy of the pre-trained SFT model ($\pi_\text{ref}$)?
**Answer**: The reference model $\pi_\text{ref}$ serves as an anchor. The term $\log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)}$ measures how much the policy shifts relative to baseline. Without $\pi_\text{ref}$, the model cannot compute implicit rewards or enforce KL regularization.

---

## 2. Capability Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands why DPO is simpler and more stable than PPO |
| **Competent** | Can derive implicit rewards $\hat{r}(x, y) = \beta (\log \pi_\theta - \log \pi_\text{ref})$ and implement `dpo_loss` |
| **Master** | Can build a complete `DPOAlignmentTrainer`, monitor reward margin convergence, and verify reference model invariance |

# Module 021: Direct Preference Optimization (DPO) & Alignment

> "Traditional RLHF requires training a separate Reward Model, an Actor Model, a Critic Model, and a Reference Model using PPO reinforcement learning—a pipeline notorious for hyperparameter instability. Direct Preference Optimization (DPO), introduced by Rafailov et al. (2023), mathematically proves that the optimal policy can be trained DIRECTLY on preference pairs $(x, y_w, y_l)$ using a simple binary cross-entropy loss."

---

## 1. Motivation: From PPO to DPO

In standard RLHF (InstructGPT):
1. **Step 1**: Supervised Fine-Tuning (SFT).
2. **Step 2**: Train a scalar Reward Model $r_\phi(x, y)$ on human preference pairs using Bradley-Terry loss.
3. **Step 3**: Fine-tune policy $\pi_\theta$ using Proximal Policy Optimization (PPO) with KL penalty relative to $\pi_\text{ref}$.

**Problems with PPO**:
- Complex 4-model setup in GPU memory (Actor, Critic, Reward, Reference).
- Mode collapse, reward hacking, and extreme training instability.

**The DPO Innovation**:
Rafailov et al. showed that the reward function $r(x, y)$ can be re-parameterized exactly in terms of the language model policy $\pi_\theta$:

$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)}$$

Substituting this implicit reward into the Bradley-Terry preference model yields the **DPO Loss**:

$$\mathcal{L}_\text{DPO}(\theta; \pi_\text{ref}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)} \right) \right]$$

---

## 2. PPO vs DPO Architectural Comparison

| Metric | PPO (RLHF) | Direct Preference Optimization (DPO) |
|---|---|---|
| **Models in Memory** | 4 Models (Actor, Critic, Reward, Ref) | **2 Models** (Policy $\pi_\theta$, Ref $\pi_\text{ref}$) |
| **Reward Model Training** | Required (Separate phase) | **Not Needed** (Implicit reward) |
| **Training Stability** | Unstable (RL policy gradients, GAE) | **Extremely Stable** (Supervised Binary Cross-Entropy) |
| **Compute Overhead** | High ($4\times$ model VRAM) | Low ($2\times$ model VRAM) |

---

## 3. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → The tug-of-war rope & implicit reward scale
03-mathematics.md       → Bradley-Terry preference derivation & DPO gradient analysis
04-implementation.py    → DPOLoss, ImplicitRewardCalculator, DPOTrainerStep
05-experiments.py       → Implicit reward margin convergence & beta hyperparameter sweep
06-real-applications.md → Alignment pipelines (Zephyr-7B, LLaMA 3 Instruct, DPO vs KTO)
07-engineering-challenge.md → Preference Preference Alignment Training Loop
08-assessment.md        → Readiness check
09-references.md        → Rafailov et al. (2023)
```

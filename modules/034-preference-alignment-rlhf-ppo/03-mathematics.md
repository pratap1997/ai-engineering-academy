# Mathematics of Preference Alignment

## 1. The Bradley-Terry Model
How do we model human preferences? The Bradley-Terry model assumes that the probability of choosing response $y_w$ over $y_l$ for prompt $x$ is dictated by the difference in their latent rewards $r(x, y)$:

$$ P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l)) $$

where $\sigma$ is the sigmoid function: $\sigma(z) = \frac{1}{1 + e^{-z}}$.

## 2. RLHF and PPO
In standard RLHF, we first train a Reward Model $r_\phi(x, y)$ using a cross-entropy loss over the Bradley-Terry probabilities:

$$ \mathcal{L}_R = - \mathbb{E}_{(x, y_w, y_l)} [\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))] $$

Then, we optimize the policy $\pi_\theta$ to maximize this reward while minimizing the Kullback-Leibler (KL) divergence from the reference policy $\pi_{ref}$:

$$ \max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta \mathbb{D}_{KL}(\pi_\theta(y \mid x) \parallel \pi_{ref}(y \mid x)) \right] $$

This objective is typically maximized using Proximal Policy Optimization (PPO), which uses a clipped objective to prevent destructively large updates:

$$ \mathcal{L}^{CLIP}(\theta) = \mathbb{E} \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right] $$

## 3. Direct Preference Optimization (DPO)
DPO introduces a mathematical sleight-of-hand. The authors noted that the optimal policy $\pi^*$ for the KL-constrained RL problem has a closed-form solution:

$$ \pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{ref}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right) $$

By rearranging this equation, we can express the reward $r(x,y)$ directly in terms of the policy and the reference policy:

$$ r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \log Z(x) $$

Substituting this implicit reward back into the Bradley-Terry model, the partition function $Z(x)$ cancels out! This gives the DPO loss:

$$ \mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)} \right) \right] $$

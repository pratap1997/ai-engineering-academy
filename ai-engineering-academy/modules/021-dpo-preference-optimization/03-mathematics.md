# Module 021: Mathematics — Bradley-Terry Preference Derivation & DPO Loss

## 1. Bradley-Terry Preference Model

Given prompt $x$ and pair of completions $(y_w, y_l)$, human preference probability under true reward function $r^*(x, y)$ is modeled by Bradley-Terry:

$$p^*(y_w \succ y_l \mid x) = \sigma(r^*(x, y_w) - r^*(x, y_l)) = \frac{1}{1 + \exp(-(r^*(x, y_w) - r^*(x, y_l)))}$$

---

## 2. Derivation of DPO Implicit Reward

The constrained RLHF objective under reference model $\pi_\text{ref}$ with KL penalty multiplier $\beta$ is:

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot \mid x)} [r(x, y)] - \beta D_{\text{KL}}(\pi(y \mid x) \parallel \pi_\text{ref}(y \mid x))$$

Setting the derivative of the Lagrangian to zero yields the exact closed-form optimal policy solution:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_\text{ref}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

Taking log on both sides and solving for $r(x, y)$:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta \log Z(x)$$

---

## 3. The DPO Loss Function

Substituting $r(x, y)$ into the Bradley-Terry preference probability (where partition function $\beta \log Z(x)$ cancels out!):

$$p^*(y_w \succ y_l \mid x) = \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)} \right)$$

Taking negative log-likelihood over dataset $\mathcal{D}$:

$$\mathcal{L}_\text{DPO}(\theta; \pi_\text{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)} \right) \right]$$

### Implicit Reward Definition:
$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)}$$

$$\text{Margin} = \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$$

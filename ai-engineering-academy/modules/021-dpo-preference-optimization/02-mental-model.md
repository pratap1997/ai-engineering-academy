# Module 021: Mental Model — The Tug-of-War & Implicit Reward

## 1. The Tug-of-War Analogy

Imagine a judge evaluating two candidate essays submitted for a prompt $x$:
- Candidate $y_w$: Preferred (Winner) essay.
- Candidate $y_l$: Dispreferred (Loser) essay.

- **PPO Alignment**: Hiring a referee (Reward Model) who gives points, then coaching the writer through trial and error (Reinforcement Learning) while trying not to change the writer's underlying voice too much (KL penalty).
- **DPO Alignment**: The writer compares how much *more likely* they are to produce $y_w$ vs $y_l$ compared to their starting baseline ($\pi_\text{ref}$). The loss pulls up the probability of $y_w$ relative to baseline and pushes down the probability of $y_l$ relative to baseline.

```
DPO Implicit Reward Margin:
                       [               Margin (r_w - r_l)               ]
               Baseline π_ref                                    Policy π_θ
                      │                                             │
      Push Down <─────┴─────> Pull Up               Push Down <─────┴─────> Pull Up
    Prob(y_l|x)              Prob(y_w|x)          Prob(y_l|x)              Prob(y_w|x)
```

---

## 2. Gradient Dynamics (Self-Weighting Loss)

The gradient of DPO loss with respect to policy parameters $\theta$ is:

$$\nabla_\theta \mathcal{L}_\text{DPO} = -\beta \cdot \underbrace{\sigma\left(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\right)}_{\text{Weighting factor } w(x, y_w, y_l)} \left[ \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right]$$

- **When model is wrong** ($\hat{r}(y_l) > \hat{r}(y_w)$): The weighting factor $w \approx 1$. Gradient pushes **hard** to boost $y_w$ and suppress $y_l$.
- **When model is right** ($\hat{r}(y_w) \gg \hat{r}(y_l)$): The weighting factor $w \approx 0$. Gradient automatically **vanishes**, preventing over-fitting!

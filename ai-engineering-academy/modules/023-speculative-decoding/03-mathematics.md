# Module 023: Mathematics — Rejection Sampling & Distribution Recovery

## 1. Speculative Sampling Acceptance Criterion

Let $q(x)$ be the probability of token $x$ under the Draft Model.
Let $p(x)$ be the probability of token $x$ under the Target Model.

Draw a uniform random scalar $r \sim U(0, 1)$.

Token $x$ is **accepted** if and only if:

$$r \le \min\left(1, \frac{p(x)}{q(x)}\right)$$

- If $p(x) \ge q(x)$: Acceptance ratio $\frac{p(x)}{q(x)} \ge 1 \implies$ Always accepted!
- If $p(x) < q(x)$: Accepted with probability $\frac{p(x)}{q(x)}$.

---

## 2. Rejection Resampling Distribution

If token $x$ is rejected, sample replacement token $x'$ from adjusted probability distribution $p'(x)$:

$$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{z} \max(0, p(z) - q(z))}$$

---

## 3. Mathematical Proof of Exact Target Recovery

We prove that the probability of accepting $x$ via speculative sampling matches $p(x)$ exactly:

$$\mathbb{P}(\text{Output } x) = \mathbb{P}(\text{Draft proposes } x \text{ AND } x \text{ accepted}) + \mathbb{P}(\text{Draft token rejected}) \cdot p'(x)$$

1. First term:
$$\mathbb{P}(\text{Draft } x \text{ AND } x \text{ accepted}) = q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) = \min(q(x), p(x))$$

2. Probability of rejection across all tokens:
$$\mathbb{P}(\text{Rejection}) = \sum_{z} q(z) \left(1 - \min\left(1, \frac{p(z)}{q(z)}\right)\right) = \sum_{z} \max(0, q(z) - p(z))$$

Notice that $\sum_z (p(z) - q(z)) = 1 - 1 = 0 \implies \sum_z \max(0, p(z) - q(z)) = \sum_z \max(0, q(z) - p(z))$.

3. Multiplying second term:
$$\mathbb{P}(\text{Rejection}) \cdot p'(x) = \sum_{z} \max(0, q(z) - p(z)) \cdot \frac{\max(0, p(x) - q(x))}{\sum_{z} \max(0, p(z) - q(z))} = \max(0, p(x) - q(x))$$

4. Total Probability:
$$\mathbb{P}(\text{Output } x) = \min(q(x), p(x)) + \max(0, p(x) - q(x)) = p(x)$$

**Q.E.D.**: Speculative Decoding produces the **exact Target distribution $p(x)$** with zero approximation error!

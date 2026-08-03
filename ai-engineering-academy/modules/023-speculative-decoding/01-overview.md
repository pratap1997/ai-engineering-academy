# Module 023: Speculative Decoding & Draft Model Verification

> "Autoregressive generation in LLMs is memory-bandwidth bound: generating 1 token requires loading all 70B parameters ($140\text{ GB}$) into GPU compute cores, utilizing $<1\%$ of FLOPS capacity. Speculative Decoding (Leviathan et al., 2023) uses a tiny 1B Draft model to guess $K$ tokens ahead, then uses the 70B Target model to verify all $K$ tokens in a **SINGLE parallel forward pass**, speeding up LLM inference by $2\times$--$3\times$ with ZERO loss in generation quality!"

---

## 1. Motivation: The Autoregressive Bottleneck

When generating text token-by-token:
- Time to generate 1 token with 70B model: $\sim 30\text{ ms}$ (loading 140 GB weights over 5 TB/s memory bus).
- Time to generate 5 tokens sequentially: $5 \times 30 = \mathbf{150\text{ ms}}$.

**Speculative Decoding Loop**:
1. **Draft Phase**: A small 1B model generates $K=5$ candidate tokens sequentially in $5 \times 2 = 10\text{ ms}$.
2. **Verification Phase**: The large 70B model evaluates all 5 candidate tokens in **1 parallel forward pass** ($30\text{ ms}$).
3. **Accept/Reject**: Rejection sampling accepts $M \le 5$ tokens and generates 1 bonus token.
4. Total time for up to 6 tokens: $10 + 30 = \mathbf{40\text{ ms}}$ ($3.75\times$ faster!).

---

## 2. Rejection Sampling Probability Guarantee

To guarantee that Speculative Decoding produces the **exact same probability distribution** as sampling directly from the Target model:

For candidate token $x$ proposed by Draft model $M_q$ and evaluated by Target model $M_p$:

$$\text{Acceptance Probability: } P_\text{accept}(x) = \min\left(1, \frac{p(x)}{q(x)}\right)$$

If token $x$ is rejected at step $i$, sample replacement token $x'$ from adjusted distribution:

$$p'(x) = \text{relu}(p(x) - q(x)) = \max(0, p(x) - q(x))$$

This mathematically guarantees that the output distribution is **100% identical** to running the 70B Target model alone!

---

## 3. Standard Decoding vs Speculative Decoding

| Metric | Standard Sequential Decoding | Speculative Decoding ($K=5$) |
|---|---|---|
| **70B Forward Passes per Token** | 1 Pass per token | $\sim 0.25$ Passes per token |
| **GPU Arithmetic Intensity** | Low ($<1\%$ Memory-bound) | High ($3\times$--$5\times$ Compute-bound) |
| **Wall-Clock Latency** | $30\text{ ms}$ / token | **$10\text{--}12\text{ ms}$ / token** |
| **Output Quality** | Baseline Target Distribution | **100% Identical to Target Distribution** |

---

## 4. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → The executive assistant draft & CEO approval
03-mathematics.md       → Rejection sampling acceptance condition & distribution recovery proof
04-implementation.py    → DraftGenerator, SpeculativeVerifier, RejectionSampler
05-experiments.py       → Acceptance rate K sweep & latency speedup benchmark
06-real-applications.md → vLLM Speculative Decoding, Medusa heads, Eagle-2
07-engineering-challenge.md → Speculative Text Generator with KV Cache Rollback
08-assessment.md        → Readiness check
09-references.md        → Leviathan et al. (2023), Chen et al. (2023)
```

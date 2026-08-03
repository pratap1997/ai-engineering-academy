# Module 023: Mental Model — Executive Assistant & CEO Approval

## 1. The Executive Assistant Analogy

Imagine a busy CEO (The 70B Target Model) writing a complex report:

- **Standard Decoding**: The CEO writes every single word line-by-line. They spend 5 minutes reading through the entire document to write just one new word.
- **Speculative Decoding**: A fast Executive Assistant (The 1B Draft Model) writes out 5 candidate sentences in a rough draft. The CEO glances at the 5-word draft all at once, approves the first 4 words, corrects the 5th word, and continues! The CEO saved **80% of their time**!

```
Speculative Decoding Verification Loop:
[Draft Model 1B]  ===> Proposes K=5 Tokens: ["The", "capital", "of", "France", "is"]
                             │
                             ▼
[Target Model 70B] ===> Parallel Evaluation Pass:
                         Token 1 ("The"):     Accept (p/q = 1.0)
                         Token 2 ("capital"): Accept (p/q = 1.0)
                         Token 3 ("of"):      Accept (p/q = 1.0)
                         Token 4 ("France"):  Accept (p/q = 0.95)
                         Token 5 ("is"):      Reject! (p/q = 0.20)
                             │
                             ▼
Result: Accept 4 tokens + 1 Target replacement token = 5 tokens written in 1 Target forward pass!
```

---

## 2. Rejection Sampling Guarantee

Why does rejecting tokens not degrade quality?

If the Assistant (Draft model) proposes a word that the CEO (Target model) agrees with ($p(x) \ge q(x)$), the CEO accepts it $100\%$ of the time.

If the Assistant proposes a word the CEO thinks is unlikely ($p(x) < q(x)$), the CEO accepts it with probability $\frac{p(x)}{q(x)}$. If rejected, the CEO rewrites that word using their own probability distribution!

The output text is **statistically indistinguishable** from having the CEO write every word alone.

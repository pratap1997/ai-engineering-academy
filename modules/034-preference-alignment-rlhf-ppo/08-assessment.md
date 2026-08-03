# Assessment

1. **What is the primary purpose of the Reward Model in PPO-based RLHF?**
   - **Answer:** To act as a proxy for human preference, providing a scalar reward for any given prompt and response.

2. **Why does PPO use a clipped objective?**
   - **Answer:** To prevent destructively large updates to the policy by keeping the new policy close to the old policy in probability space.

3. **What is the role of the KL divergence penalty?**
   - **Answer:** It penalizes the policy for drifting too far from the reference model (the base/SFT model) to prevent mode collapse or "reward hacking".

4. **How does DPO simplify the RLHF pipeline?**
   - **Answer:** DPO eliminates the need to explicitly train a Reward Model and avoids the unstable Reinforcement Learning (PPO) loop.

5. **In DPO, what serves as the implicit reward?**
   - **Answer:** The scaled log-ratio of the policy model's probability over the reference model's probability.

6. **What is the Bradley-Terry model used for?**
   - **Answer:** It models the probability that one item is preferred over another based on the difference of their underlying scalar rewards.

7. **What is the primary failure mode (reward hacking) in RLHF?**
   - **Answer:** The policy model learns to exploit flaws in the Reward Model to achieve high scores without actually producing desirable output (e.g., repeating the same highly-scored word).

8. **In the DPO loss function, what does the $\beta$ parameter control?**
   - **Answer:** The strength of the KL divergence penalty; a higher $\beta$ forces the policy to stay closer to the reference model.

9. **If PPO is more complex, why is it still used?**
   - **Answer:** In some empirical settings, PPO handles out-of-distribution generation better because it explores the action space actively during training, whereas DPO only learns from the static pairs in the dataset.

10. **What is RLAIF?**
    - **Answer:** Reinforcement Learning from AI Feedback. It uses a strong LLM (instead of humans) to generate the preference labels or rewards based on a rubric or constitution.

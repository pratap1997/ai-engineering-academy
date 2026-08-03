# Engineering Challenge: Evaluator Suite

**Objective:**
Build a Preference Alignment Evaluator that compares PPO, DPO, and KTO (Kahneman-Tversky Optimization) loss functions.

**Requirements:**
1. Implement a mock `KTOLossCalculator`.
2. Given a preference dataset of 100 pairs, compute the trajectory of loss across all three methods if the policy incrementally shifts 10% closer to the preference in each step.
3. Your solution must not use PyTorch or TensorFlow, only the standard math library.
4. Output a summary report of the loss scaling.

*No hints provided.*

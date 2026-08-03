# Module 034: Preference Alignment & RLHF (PPO vs DPO)

## Overview
Pre-training on vast amounts of internet text produces language models capable of predicting the next token, but they do not inherently know how to act as helpful, harmless, and honest assistants. **Preference Alignment** bridges this gap, transforming a base model into an aligned assistant.

This module explores the two dominant paradigms of preference alignment:
1. **Reinforcement Learning from Human Feedback (RLHF) with PPO:** The classic three-stage pipeline (Supervised Fine-Tuning -> Reward Modeling -> Policy Optimization).
2. **Direct Preference Optimization (DPO):** A simplified, mathematically elegant approach that eliminates the separate reward model and RL loop by optimizing preferences directly on the language model's policy.

## Motivation
Why do we need alignment?
- **Base models hallucinate, refuse arbitrarily, and generate toxic content.**
- Pre-training optimizes for *likelihood*, not *desirability*.
- Without alignment, the user must meticulously prompt-engineer to coax the desired behavior.

## The Core Problem
Given a prompt $x$, a preferred response $y_w$ (win), and a rejected response $y_l$ (loss), how do we update the model's weights to increase the probability of $y_w$ relative to $y_l$ without destroying its foundational knowledge?

In this module, you will learn how to implement the math behind both PPO's reward modeling and DPO's direct loss formulation from scratch in Python.

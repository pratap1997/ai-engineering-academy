# Module 022: Mental Model — The Master Professor & Apprentice

## 1. The Professor and Apprentice Analogy

Imagine a student studying for a medicine exam:

- **Hard Label Training**: Reading an answer key that only says: "Question 1: Choice A is correct". The student memorizes A without understanding why Choice B was a close second choice or why Choice D was completely wrong.
- **Knowledge Distillation**: Sitting next to a master professor who explains: "Choice A is correct (85% confidence), but Choice B is an understandable confusion (12% confidence) because of similar symptoms, while Choice D is impossible (0.01% confidence)". The student learns the **reasoning landscape** of the domain!

```
Hard One-Hot Label:     [ 1.00,  0.00,  0.00,  0.00 ]  (No dark knowledge)
Teacher Softmax (T=1):  [ 0.98,  0.019, 0.001, 0.000 ] (Hard to see small probabilities)
Teacher Softmax (T=4):  [ 0.45,  0.30,  0.15,  0.10 ]  (Rich dark knowledge exposed!)
```

---

## 2. Temperature Scaling Intuition

At temperature $T=1$, the softmax function acts as a sharp peak generator, squashing smaller probabilities near zero:

$$q_i = \frac{e^{z_i}}{e^{z_i} + e^{z_j} + e^{z_k}}$$

When temperature is raised to $T=4$, the exponent is softened ($e^{z_i / 4}$), flattening the distribution and amplifying secondary and tertiary probabilities so the student can learn sub-class structural similarities!

# 01 — Overview

## What problem does a perceptron solve?

Imagine you are trying to teach a machine to make a simple yes/no decision.

Not a complex one. Not "is this a cat?" or "will this patient survive?"  
Something simpler: "given two numbers, is their combination above a threshold?"

The perceptron is the simplest possible learning machine for this kind of problem.  
It takes a set of numerical inputs, weighs each one by importance, adds a bias,
and outputs a binary decision: **0 (no) or 1 (yes)**.

It does not use probability. It does not use layers. It does not use backpropagation.  
It uses a single rule: **if the weighted sum is at or above zero, predict 1. Otherwise, predict 0.**

And it *learns* — not by being told the answer, but by updating its weights whenever it gets one wrong.

---

## Historical context

In 1957, Frank Rosenblatt at Cornell Aeronautical Laboratory built the **Mark I Perceptron** —
a hardware machine (not software) designed to recognize simple geometric shapes.

It had 400 photocell inputs connected to 512 association units, with potentiometers
that adjusted weights *physically* during training. It was built to learn.

The New York Times called it:
> *"the embryo of an electronic computer that the Navy expects will be able to walk, talk,
> see, write, reproduce itself and be conscious of its existence."*

That was overblown. But the core idea was not.

In 1969, Minsky and Papert published *Perceptrons*, which proved mathematically that
a single perceptron cannot solve XOR — it cannot separate data that is not linearly separable.  
This caused a sharp reduction in AI funding and interest: the **first AI winter**.

The perceptron was not useless. It was misunderstood as more capable than it was.  
**That is the engineering lesson, not just the history.**

Understanding what a model *cannot* do is as important as knowing what it can.

---

## Why learn this first?

The perceptron is the atomic unit of every modern neural network.

One neuron in a deep neural network is doing the same computation:
```
weighted sum → nonlinearity → output
```

The difference is scale, depth, and the nonlinearity used.

Learn this once, correctly, and you will never need to re-learn it.
Every subsequent concept — loss functions, gradient descent, multilayer networks,
attention mechanisms — builds on this foundation.

---

## Learning outcomes

By the end of this module you can:

1. Explain how weighted inputs, bias, and a threshold produce a binary decision.
2. Derive the prediction rule from first principles.
3. Derive the update rule from first principles.
4. Implement a perceptron in plain Python without libraries.
5. Implement an equivalent NumPy version.
6. Train it on AND and OR.
7. Demonstrate and explain why it fails on XOR.
8. Visualize the linear decision boundary.
9. Explain when a perceptron should not be used.

---

## When a perceptron should NOT be used

This is the question most courses skip. Do not skip it.

| Situation | Why the perceptron fails |
|---|---|
| XOR and any non-linearly-separable data | Perceptron can only learn linear boundaries |
| Multi-class classification | Binary output only; multi-class requires extensions |
| Noisy labels or overlapping classes | No probabilistic output; cannot express confidence |
| Complex real-world data (images, text, audio) | Single layer is far too shallow |
| Any task where the decision boundary is curved | Cannot represent non-linear relationships |

These are not edge cases. Most real-world problems fall into one or more of these categories.

The perceptron's value is as a **foundational component** and a **learning instrument**.
Its value in modern production systems is nearly zero in isolation.

That honest assessment is what makes this module useful.

---

## This module does NOT cover

- Multilayer perceptrons (Module 004)
- Backpropagation (Module 004)
- Loss functions and gradient descent (Modules 002–003)
- PyTorch, TensorFlow, or scikit-learn
- Multiclass classification
- Production deployment

If you see those topics in an introduction to perceptrons elsewhere, that course
is teaching too much at once — and teaching none of it deeply enough.

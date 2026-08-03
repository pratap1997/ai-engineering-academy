# 08 — Assessment

Read this artifact **after** completing the engineering challenge.  
The answer keys are here. Do not read them before your attempt.

---

## Part A — Concept Checks

Answer each question before revealing the key.

---

**A1.** What is the role of the bias term in a perceptron?

<details>
<summary>Answer key</summary>

The bias shifts the decision boundary independently of the input values.
Without bias, the decision boundary must pass through the origin (all zeros),
severely limiting what problems the perceptron can solve.

With bias, the boundary can be positioned anywhere in feature space —
it controls the perceptron's default tendency before seeing any input.

**Common wrong answer:** "The bias scales the inputs."  
**Why it is wrong:** The bias is not connected to any input. It is a scalar added
directly to the pre-activation. It shifts the threshold, not the inputs.
</details>

---

**A2.** What are the three possible values of the error signal delta = y − ŷ, and what does each mean?

<details>
<summary>Answer key</summary>

| delta | y | ŷ | Meaning |
|---|---|---|---|
| 0 | 1 | 1 | Correct — no update |
| 0 | 0 | 0 | Correct — no update |
| +1 | 1 | 0 | Missed positive — increase weights |
| −1 | 0 | 1 | False positive — decrease weights |

Weights and bias are only updated when delta ≠ 0.
This is what makes the perceptron an *error-driven* learner.

**Common wrong answer:** "The perceptron updates on every example."  
**Why it is wrong:** When delta = 0, the update rule produces 0 change.
Updating only on mistakes is both correct and computationally efficient.
</details>

---

**A3.** The perceptron is trained with learning_rate = 0.0. What happens to the weights?

<details>
<summary>Answer key</summary>

Nothing. The update rule is `w += η * delta * x`, and when η = 0, no update occurs
regardless of the error. The weights stay at their initial values (typically zero),
and the perceptron never learns.

This is tested in `tests/test_perceptron.py` as an edge case.

**Common wrong answer:** "It would converge faster."  
**Why it is wrong:** There is no update at all. The learning rate controls the step size —
zero step size means zero learning.
</details>

---

**A4.** What is the Perceptron Convergence Theorem, and what is its practical significance?

<details>
<summary>Answer key</summary>

The theorem states: if training data is linearly separable and inputs are bounded,
the perceptron is guaranteed to find a solution in a finite number of updates.

Practical significance: the theorem makes convergence on AND and OR mathematically
certain (given enough epochs), not just empirically observed.

The theorem also reveals its own boundary: it says nothing about non-linearly-separable
data. On XOR, no such guarantee exists — and in fact, no solution exists.

**Common wrong answer:** "It guarantees the perceptron will always converge."  
**Why it is wrong:** Convergence is only guaranteed for linearly separable data.
The theorem does not apply to XOR.
</details>

---

**A5.** If you increase the number of epochs from 100 to 10,000 on XOR data, will the perceptron eventually learn XOR?

<details>
<summary>Answer key</summary>

No. The number of epochs does not change the fundamental geometric limitation.
XOR is not linearly separable — no single straight line can separate its positive
and negative examples in 2D space. More training will not change that geometry.

The perceptron will continue oscillating between states without converging.

**Common wrong answer:** "With enough epochs, it might find a boundary that works."  
**Why it is wrong:** The theorem states convergence is only possible when a correct
weight vector *exists*. For XOR, no such vector exists for a single-layer perceptron.
10,000 epochs on an impossible task produces 10,000 failed updates.
</details>

---

## Part B — Debugging Questions

---

**B1.** A perceptron is trained on AND data and consistently predicts 0 for all inputs, including [1, 1]. The training loop runs without errors. What are the three most likely causes?

<details>
<summary>Answer key</summary>

**Cause 1: Weights and bias are never updated.**  
Check that the update fires when `delta != 0`, not when `delta == 0`.
A common bug is inverting this condition: `if delta == 0: update`.

**Cause 2: Learning rate is zero or very close to zero.**  
Even if the condition is correct, a near-zero learning rate produces near-zero updates.
After 100 epochs, weights are still essentially zero → predict 0 for everything.

**Cause 3: The bias is not included in the prediction.**  
If the pre-activation is computed as `z = dot(w, x)` without adding `b`,
the decision boundary is forced through the origin.
For AND data starting with all-zero weights, this means all predictions stay 0.

**Diagnostic approach:** Add print statements inside the training loop.
After epoch 1, are the weights changing? Are they changing in the right direction?
</details>

---

**B2.** A learner's perceptron reaches 100% accuracy on AND and OR, but the tests fail with:  
`AssertionError: predict() must return only 0 or 1`  
What is wrong?

<details>
<summary>Answer key</summary>

The `predict()` method is returning float values (e.g., `0.0` and `1.0`) instead of integers.

In Python, `0.0 in (0, 1)` evaluates to `True` due to float-int equality, but
the strict test `v in (0, 1)` may behave differently depending on the implementation.

More likely: the method returns a numpy array with dtype float64, or returns
values like `True` and `False` from a comparison operation.

**Fix:** Cast the output to int: `return int(z >= 0)` or `return (z >= 0).astype(int)`.
</details>

---

**B3.** A learner's perceptron learns AND correctly in training but fails when `predict()` is called on new data. The model was fit on a Python list, but `predict()` is called with a NumPy array. What could cause this?

<details>
<summary>Answer key</summary>

The inner loop in `predict()` may rely on Python list indexing behavior that does
not transfer to NumPy arrays — for example, using `x[j]` where `x` is a 1D NumPy
array works the same, but iterating with `zip(weights, x)` on a NumPy array
iterates over elements, which should also work.

More likely: the `fit()` method stored `self.n_features = len(X[0])` where `X`
was a Python list, giving an integer. When called with a NumPy array,
`len(X[0])` on a 1D NumPy array also works. However, `len(X)` on a 2D NumPy
array gives the number of rows, not the number of features.

**Best fix:** Convert inputs explicitly at the start of both `fit()` and `predict()`.
Defensive conversion prevents this entire class of bugs.
</details>

---

## Part C — Explain Like an Engineer

---

**C1.** A junior engineer on your team says: "I tried training a perceptron on our dataset and it won't converge. I've run it for 10,000 epochs. How do I fix it?"

Write the first three questions you would ask before suggesting a solution.

<details>
<summary>Answer key</summary>

**Question 1:** Is the data linearly separable?  
Plot it (if 2D) or run a quick logistic regression. If the linear classifier fails,
the problem is not the perceptron's parameters — the data requires a nonlinear model.

**Question 2:** What does "won't converge" mean specifically?  
Is accuracy oscillating? Is it stuck at a fixed wrong answer? Is it improving slowly?
"Won't converge" can mean many different things with different causes.

**Question 3:** Have you checked for class imbalance?  
If 99% of examples are class 0, a perceptron may learn to always predict 0 and achieve
99% accuracy without actually learning. This looks like convergence but is not.

Only after these answers can you give a useful recommendation.
The most common mistake is jumping to "increase epochs" or "change learning rate"
before diagnosing whether the problem is the model or the data.
</details>

---

**C2.** Explain to a non-technical colleague what the perceptron is doing, using only the weighted voting analogy. Do not use any equations.

<details>
<summary>Answer key (one acceptable version)</summary>

"Imagine a committee of experts, each voting yes or no on a decision.
Each expert's vote is weighted by how reliable they are — a more accurate expert
counts for more. There is also a chairperson who adds a small standing vote
for or against, regardless of what the experts say.

The committee adds up all the weighted votes plus the chairperson's vote.
If the total is positive, the decision is yes. If negative, it is no.

The perceptron learns by adjusting the weights after each wrong decision.
If it said yes but the answer was no, it reduces the influence of the experts
who argued for yes. If it said no but the answer was yes, it increases their influence.

Over time, it learns which experts to trust and how much."
</details>

---

## Misconception Record

If you discovered any misconception during this module — something you believed
before that you now know is wrong — record it here:

```
Misconception: [what you thought]
Correction:    [what is actually true]
Why it matters: [what wrong decisions this misconception would lead to]
```

These records become assessment questions, mental model improvements, and experiment ideas
for the next version of this module. See `engineering/progression-log.md`.

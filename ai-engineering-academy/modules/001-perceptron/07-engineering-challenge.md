# 07 — Engineering Challenge

## Build a Binary Perceptron

**Time expectation:** 45 minutes  
**Status:** Do this before reading `08-assessment.md`

---

## Objective

Implement a working binary perceptron from scratch.  
Do not copy or closely adapt the reference implementation in `04-implementation.py`.

Close that file. Build yours.

---

## Constraints

| Rule | Detail |
|---|---|
| **No ML libraries** | No scikit-learn, PyTorch, TensorFlow, Keras |
| **NumPy is optional** | You may use it, or use only built-in Python — your choice |
| **No copying** | Do not copy `04-implementation.py`. Use it as a conceptual reference only if truly stuck. |
| **Required interface** | Your class must have `fit(X, y)` and `predict(X)` methods |
| **Test requirement** | Your implementation must pass the acceptance tests (see below) |
| **Written explanation** | Submit a written explanation of the bias term's role (see format below) |

---

## Required Interface

```python
class MyPerceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        ...

    def fit(self, X, y):
        """
        Train on labeled examples.
        X: list or array of shape (n_samples, n_features)
        y: list or array of 0s and 1s, shape (n_samples,)
        Returns: self
        """
        ...

    def predict(self, X):
        """
        Predict labels for input samples.
        X: list or array of shape (n_samples, n_features)
        Returns: list or array of 0s and 1s
        """
        ...
```

You may add helper methods. Do not remove the required interface.

---

## Acceptance Tests

Your implementation must pass all of these:

```python
# AND gate: must reach 100% accuracy with fixed seed
AND_X = [[0,0],[0,1],[1,0],[1,1]]
AND_y = [0, 0, 0, 1]

p = MyPerceptron(learning_rate=0.1, epochs=100)
p.fit(AND_X, AND_y)
predictions = p.predict(AND_X)
assert predictions == [0, 0, 0, 1], f"AND failed: {predictions}"

# OR gate: must reach 100% accuracy
OR_X = [[0,0],[0,1],[1,0],[1,1]]
OR_y = [0, 1, 1, 1]

p = MyPerceptron(learning_rate=0.1, epochs=100)
p.fit(OR_X, OR_y)
predictions = p.predict(OR_X)
assert predictions == [0, 1, 1, 1], f"OR failed: {predictions}"

# XOR gate: must NOT reach 100% accuracy (3 seeds)
XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_y = [0, 1, 1, 0]
# No seed mechanism in this simple test — just verify < 100% accuracy

p = MyPerceptron(learning_rate=0.1, epochs=100)
p.fit(XOR_X, XOR_y)
xor_predictions = p.predict(XOR_X)
correct = sum(p == t for p, t in zip(xor_predictions, XOR_y))
assert correct < 4, f"XOR unexpectedly perfect: {xor_predictions}"

# predict() must return only 0s and 1s
assert all(v in (0, 1) for v in predictions), "predict() must return only 0 or 1"

print("All acceptance tests passed.")
```

---

## XOR Explanation Requirement

After running the XOR experiment, write a one-paragraph explanation that answers:

- **Why** does the perceptron fail on XOR?
- Is it a problem with the learning rate or epochs, or something more fundamental?
- What would need to change about the architecture to solve XOR?

This explanation is part of your submission. It should be based on geometric reasoning,
not just "XOR is not linearly separable" without explaining what that means.

---

## Bias Explanation Requirement

Write one to three sentences explaining:

> What does the bias term do, and what would happen if it were removed?

Your explanation should be concrete. Describe the geometric effect, not just the equation.

---

## Submission Format

Create a folder: `submissions/001-perceptron/your-username/`

Include:
```
solution.py          Your implementation
results.txt          Output from running the acceptance tests
explanation.md       XOR explanation + bias explanation
notes.md             What was hard, what surprised you, what you'd do differently
```

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for PR instructions.

---

## If You Get Stuck

Use this order:
1. Re-read `02-mental-model.md` — the weighted voting system description.
2. Re-read the equations in `03-mathematics.md` — translate each one to code.
3. Check that your update rule fires only on mistakes (delta ≠ 0), not every example.
4. Check that your bias is updated with the same rule as the weights.

Do not open `04-implementation.py` until you have spent at least 30 minutes on your own attempt.
Even a broken attempt is more valuable than a copied working one.

---

## Time record

Record your actual time:
- Start time: ___
- First AND test passing: ___
- All tests passing: ___
- Explanations written: ___
- Total: ___

Honest time records help calibrate the estimated time for future learners.

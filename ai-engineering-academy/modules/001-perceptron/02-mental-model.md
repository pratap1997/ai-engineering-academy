# 02 — Mental Model

## The core model: a weighted voting system

Forget equations for now.

Imagine a panel of three judges deciding whether to approve a loan.

Each judge votes for or against. But not all judges have equal authority:
- Judge A (credit score): weight = 3
- Judge B (employment history): weight = 2  
- Judge C (current debt): weight = -2 (negative — high debt argues *against*)

There is also a chairperson who has a standing bias:
the chairperson tends to approve borderline cases (bias = +1).

The panel adds up all weighted votes plus the chairperson's bias.
If the total is zero or above: approve. If below zero: reject.

**That is a perceptron.**

```
Input 1 (credit score) ──── weight 3 ──────┐
Input 2 (employment)   ──── weight 2 ──────┤
                                            ├──→ Weighted sum + bias ──→ Threshold ──→ Decision
Input 3 (debt)         ──── weight -2 ─────┤
Bias                   ──── +1 ────────────┘
```

The weights determine how much each input matters.  
The bias shifts where the threshold sits, independently of the inputs.  
The threshold produces the final binary decision.

---

## The four components

### Inputs (x)

A vector of numerical values describing one example.

```
x = [credit_score, employment_years, debt_ratio]
x = [0.8, 5.0, 0.3]
```

Inputs are fixed for a given example. The perceptron does not change them.

### Weights (w)

A vector of the same length as the inputs.  
Each weight says how much the corresponding input should influence the decision.

- Positive weight: this input argues *for* predicting 1.
- Negative weight: this input argues *for* predicting 0.
- Near-zero weight: this input barely matters.

**The weights are what the perceptron learns.**  
They start at zero (or small random values) and are adjusted after each mistake.

### Bias (b)

A single number. It is not connected to any input.  
Think of it as the perceptron's *default tendency* before seeing any input.

Without bias, the decision boundary must always pass through the origin.  
With bias, it can be shifted anywhere. This matters for almost every real problem.

> **Common misconception:** the bias changes the *inputs*.  
> **Correct model:** the bias shifts the *decision boundary*, independently of the input values.

### Threshold decision (step function)

After computing the weighted sum plus bias:

- If the result is ≥ 0: output 1.
- If the result is < 0: output 0.

This is not smooth. It is binary. There is no "almost yes." This is both the
perceptron's simplicity and its limitation.

---

## Visual: the decision boundary

For a two-input perceptron, the decision boundary is a straight line.

```
x₂ (input 2)
    │         o o
    │       o   o
    │     o     o
    │     ─────────── decision boundary (w₁x₁ + w₂x₂ + b = 0)
    │   ×   ×
    │ ×   ×
    │───────────────── x₁ (input 1)

  o = predict 1
  × = predict 0
```

The perceptron's job is to find the position and angle of that line  
so that all 1s are on one side and all 0s are on the other.

**This is why XOR fails.** XOR's positive and negative examples cannot be separated
by a single straight line — they require a curved boundary, which a single perceptron
cannot produce.

```
XOR:           AND:
  0 1             0 0
  1 0             0 1

  No single line can     One line easily
  separate 1s from 0s    separates 1 from 0
```

---

## How it learns: the update rule in plain language

1. Show the perceptron one example.
2. Ask it to predict: 0 or 1?
3. If correct: do nothing. Weights stay the same.
4. If wrong:
   - If it predicted 0 but the answer was 1: *increase* the weights for active inputs.
   - If it predicted 1 but the answer was 0: *decrease* the weights for active inputs.
5. Repeat for all examples. Repeat many times (epochs).

The learning rate (η) controls how big each adjustment is.

- Too large: the perceptron overshoots and oscillates.
- Too small: the perceptron learns very slowly.
- Just right: it converges efficiently.

The bias is also updated in the same way, as if it were a weight connected
to a constant input of 1.

---

## The model in one diagram

```
         x₁ ─── w₁ ──┐
         x₂ ─── w₂ ──┤
         x₃ ─── w₃ ──┤── Σ(wᵢxᵢ) + b ──→ step(z) ──→ ŷ ∈ {0, 1}
         ...          │
         b  ─── 1  ──┘

Where:
  z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b   (weighted sum + bias)
  ŷ = 1  if z ≥ 0
  ŷ = 0  if z < 0
```

Hold this model. The mathematics in the next artifact is just this diagram written formally.

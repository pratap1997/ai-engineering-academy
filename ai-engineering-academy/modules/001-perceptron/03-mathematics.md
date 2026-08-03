# 03 — Mathematics

## Notation

| Symbol | Meaning |
|---|---|
| $x \in \mathbb{R}^n$ | Input vector, $n$ features |
| $w \in \mathbb{R}^n$ | Weight vector (what the perceptron learns) |
| $b \in \mathbb{R}$ | Bias scalar (what the perceptron learns) |
| $z \in \mathbb{R}$ | Pre-activation: weighted sum plus bias |
| $\hat{y} \in \{0, 1\}$ | Predicted label |
| $y \in \{0, 1\}$ | True label |
| $\eta > 0$ | Learning rate (hyperparameter, set by you) |
| $t$ | Training step index |
| $i$ | Sample index |

**Label convention:** this module uses 0 and 1 (not −1 and +1).  
This choice is explained at the end of this artifact.

---

## 1. Prediction rule

The perceptron computes a weighted sum of inputs and adds a bias:

$$z = w^T x + b = \sum_{j=1}^{n} w_j x_j + b$$

It then applies a step function to produce a binary prediction:

$$\hat{y} =
\begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}$$

This is deterministic. Given fixed weights and bias, the same input always produces the same output.

---

## 2. Error signal

The perceptron's error for one example is:

$$\delta = y - \hat{y}$$

This takes three values:

| $y$ | $\hat{y}$ | $\delta$ | Meaning |
|---|---|---|---|
| 1 | 1 | 0 | Correct — no update |
| 0 | 0 | 0 | Correct — no update |
| 1 | 0 | +1 | Missed positive — increase weights |
| 0 | 1 | −1 | False positive — decrease weights |

If $\delta = 0$, no update occurs. The rule is online: it updates immediately after each example.

---

## 3. Update rule

When the perceptron makes a mistake, it adjusts:

$$w^{(t+1)} \leftarrow w^{(t)} + \eta \cdot \delta \cdot x$$

$$b^{(t+1)} \leftarrow b^{(t)} + \eta \cdot \delta$$

Expanding with $\delta = y - \hat{y}$:

$$w^{(t+1)} \leftarrow w^{(t)} + \eta (y - \hat{y}) x$$

$$b^{(t+1)} \leftarrow b^{(t)} + \eta (y - \hat{y})$$

**Why does this make sense?**

- If $y = 1$ and $\hat{y} = 0$ (missed positive): $\delta = +1$, so weights increase in the direction of $x$.  
  This makes $w^T x$ larger, pushing future predictions toward 1 for similar inputs.

- If $y = 0$ and $\hat{y} = 1$ (false positive): $\delta = -1$, so weights decrease in the direction of $x$.  
  This makes $w^T x$ smaller, pushing future predictions toward 0 for similar inputs.

The bias update is identical: the bias behaves as a weight attached to a constant input of 1.

---

## 4. Training loop

Given a dataset $\{(x^{(i)}, y^{(i)})\}_{i=1}^{m}$ and $T$ training epochs:

$$\text{For } t = 1 \text{ to } T:$$
$$\quad \text{For each } (x^{(i)}, y^{(i)}):$$
$$\quad \quad z = w^T x^{(i)} + b$$
$$\quad \quad \hat{y} = \mathbf{1}[z \geq 0]$$
$$\quad \quad w \leftarrow w + \eta (y^{(i)} - \hat{y}) x^{(i)}$$
$$\quad \quad b \leftarrow b + \eta (y^{(i)} - \hat{y})$$

No batch processing. No gradient accumulation. One update per example.

---

## 5. Convergence theorem

**Perceptron Convergence Theorem** (Rosenblatt, 1962):

If the training data is **linearly separable**, the perceptron algorithm
is guaranteed to converge to a set of weights that correctly classifies all examples
in a finite number of steps.

Two conditions are required:
1. There exists a weight vector $w^*$ such that all examples are correctly classified with a margin $\gamma > 0$.
2. All input vectors satisfy $\|x^{(i)}\| \leq R$ for some finite $R$.

Under these conditions, the number of weight updates is bounded by:

$$k \leq \left(\frac{R}{\gamma}\right)^2$$

**What this means in practice:**
- If data is linearly separable: convergence is guaranteed.
- If data is NOT linearly separable (e.g., XOR): the algorithm never converges.  
  It will keep updating forever or oscillate between wrong states.

---

## 6. The decision boundary

Setting $z = 0$ gives the decision boundary:

$$w^T x + b = 0$$

For two inputs:

$$w_1 x_1 + w_2 x_2 + b = 0$$
$$x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}$$

This is a straight line. In $n$ dimensions, it is a hyperplane.

The perceptron can only learn **linear decision boundaries**.  
Any problem requiring a curved boundary is beyond its capacity.

---

## 7. Label convention: why 0/1 instead of −1/+1

The original Rosenblatt perceptron used +1 and −1 labels.
The Microsoft AI For Beginners lesson uses this convention too.

This module uses **0 and 1** for one reason: consistency with the modules that follow.

- Module 002 introduces **binary cross-entropy loss**, which assumes $y \in \{0, 1\}$.
- Module 003's **logistic regression** uses sigmoid output in $[0, 1]$.
- Module 004's **multilayer network** uses 0/1 targets throughout.

The mathematics is equivalent — only the error signal expression changes:

| Convention | Error signal |
|---|---|
| 0/1 labels | $\delta = y - \hat{y} \in \{-1, 0, +1\}$ |
| +1/−1 labels | $\delta = y \cdot \hat{y} \in \{-1, +1\}$ (update only on mistakes) |

Both produce the same weight adjustments on errors. The 0/1 convention is cleaner
for the curriculum that follows.

---

## Summary of equations

| Equation | Formula |
|---|---|
| Pre-activation | $z = w^T x + b$ |
| Prediction | $\hat{y} = 1$ if $z \geq 0$ else $0$ |
| Error | $\delta = y - \hat{y}$ |
| Weight update | $w \leftarrow w + \eta \delta x$ |
| Bias update | $b \leftarrow b + \eta \delta$ |

These four equations are everything a perceptron does.
The implementation in the next artifact is a direct translation of them.

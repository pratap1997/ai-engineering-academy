# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Module 001 — Experiments: Perceptron From Scratch
#
# **Purpose:** Controlled, reproducible observations of perceptron behaviour.
#
# Each experiment follows this structure:
# - Hypothesis
# - Setup
# - Prediction
# - Observation
# - Explanation
# - Engineering implication
# - Unexpected result (if any)
# - Reproducibility
#
# Run all cells top-to-bottom. Every result should be reproducible.
# If you observe something different from what is documented, please open
# a reproduction-failure issue (see CONTRIBUTING.md).

# %% [markdown]
# ## Setup: imports and shared utilities

# %%
import importlib.util
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the reference implementation
_script_dir = os.path.dirname(os.path.abspath(__file__))
_assets_dir = os.path.join(_script_dir, "assets")
os.makedirs(_assets_dir, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "implementation",
    os.path.join(_script_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PerceptronPython = _mod.PerceptronPython
PerceptronNumPy = _mod.PerceptronNumPy


def plot_decision_boundary(model, X, y, title="Decision Boundary", ax=None):
    """Plot the learned decision boundary for a 2-feature perceptron."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    preds = model.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, preds, alpha=0.3, cmap="RdBu")
    ax.contour(xx, yy, preds, levels=[0.5], colors="black", linewidths=1.5)

    colors = ["red" if label == 0 else "blue" for label in y]
    ax.scatter([xi[0] for xi in X], [xi[1] for xi in X],
               c=colors, s=80, zorder=5, edgecolors="black")

    red_patch = mpatches.Patch(color="red", alpha=0.5, label="Class 0")
    blue_patch = mpatches.Patch(color="blue", alpha=0.5, label="Class 1")
    ax.legend(handles=[red_patch, blue_patch])
    ax.set_title(title)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    return ax


# Shared data
AND_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
AND_y = [0, 0, 0, 1]

OR_X  = [[0, 0], [0, 1], [1, 0], [1, 1]]
OR_y  = [0, 1, 1, 1]

XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_y = [0, 1, 1, 0]

print("Setup complete.")

# %% [markdown]
# ---
# ## Experiment 1: AND Gate
#
# **Hypothesis:** A perceptron should be able to learn the AND function because
# AND is linearly separable — the single positive case (1,1) lies in a distinct
# region of the input space.

# %%
# Setup
p_and = PerceptronNumPy(learning_rate=0.1, epochs=100)
p_and.fit(AND_X, AND_y)

# Prediction: expect 100% accuracy
preds_and = p_and.predict(AND_X).tolist()
print(f"True labels: {AND_y}")
print(f"Predictions: {preds_and}")
print(f"Accuracy:    {p_and.score(AND_X, AND_y):.0%}")
print(f"Weights:     {p_and.weights}")
print(f"Bias:        {p_and.bias:.4f}")

# Visualization
fig, ax = plt.subplots(figsize=(5, 4))
plot_decision_boundary(p_and, AND_X, AND_y, title="AND Gate — Learned Boundary", ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(_assets_dir, "exp1-and-gate.png"), dpi=100)
plt.close()

# Reproducibility: seed=N/A (deterministic with fixed order), lr=0.1, epochs=100

# %% [markdown]
# **Observation:** _[fill in after running]_
#
# **Explanation:** AND is linearly separable. A straight line can separate (1,1)
# from (0,0), (0,1), and (1,0). The perceptron convergence theorem guarantees
# a solution exists and will be found.
#
# **Engineering implication:** For simple linearly separable problems, a perceptron
# is computationally efficient and provably correct.
#
# **Unexpected result:** _[fill in if anything differed from prediction]_

# %% [markdown]
# ---
# ## Experiment 2: OR Gate

# %%
p_or = PerceptronNumPy(learning_rate=0.1, epochs=100)
p_or.fit(OR_X, OR_y)

preds_or = p_or.predict(OR_X).tolist()
print(f"True labels: {OR_y}")
print(f"Predictions: {preds_or}")
print(f"Accuracy:    {p_or.score(OR_X, OR_y):.0%}")
print(f"Weights:     {p_or.weights}")
print(f"Bias:        {p_or.bias:.4f}")

fig, ax = plt.subplots(figsize=(5, 4))
plot_decision_boundary(p_or, OR_X, OR_y, title="OR Gate — Learned Boundary", ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(_assets_dir, "exp2-or-gate.png"), dpi=100)
plt.close()

# %% [markdown]
# ---
# ## Experiment 3: XOR Failure (Geometric Limitation)
#
# **Hypothesis:** The perceptron will fail to achieve 100% accuracy on XOR.
# This is not a training failure — it is a geometric impossibility.
# XOR's positive examples (0,1) and (1,0) cannot be separated from its
# negative examples (0,0) and (1,1) by a single straight line.

# %%
results = []
for seed_offset in [0, 7, 42]:
    rotated_X = XOR_X[seed_offset % 4:] + XOR_X[:seed_offset % 4]
    rotated_y = XOR_y[seed_offset % 4:] + XOR_y[:seed_offset % 4]

    p = PerceptronNumPy(learning_rate=0.1, epochs=200)
    p.fit(rotated_X, rotated_y)
    acc = p.score(XOR_X, XOR_y)
    results.append((seed_offset, acc, p.weights.tolist(), p.bias))
    print(f"seed_offset={seed_offset}: accuracy={acc:.0%}, weights={p.weights}, bias={p.bias:.4f}")

print("\nAll runs below 100%:", all(r[1] < 1.0 for r in results))

# Visualize one XOR attempt
p_xor = PerceptronNumPy(learning_rate=0.1, epochs=200)
p_xor.fit(XOR_X, XOR_y)

fig, ax = plt.subplots(figsize=(5, 4))
plot_decision_boundary(p_xor, XOR_X, XOR_y, title="XOR — No Linear Boundary Possible", ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(_assets_dir, "exp3-xor-failure.png"), dpi=100)
plt.close()

# %% [markdown]
# **Observation:** _[fill in after running]_
#
# **Explanation:** XOR is not linearly separable. Positive examples (0,1) and (1,0)
# are diagonal from each other, as are negative examples (0,0) and (1,1).
# No single straight line can put both positives on one side and both negatives
# on the other. More epochs, different learning rates — none of this helps.
# The geometry is the constraint.
#
# **Engineering implication:** Before applying any linear classifier, check whether
# the problem is linearly separable. If not, a nonlinear model (MLP, kernel SVM,
# decision tree) is required.
#
# **Unexpected result:** _[fill in]_

# %% [markdown]
# ---
# ## Experiment 4: Remove the Bias Term
#
# **Hypothesis:** Removing the bias will prevent the perceptron from learning AND
# correctly because AND requires a decision boundary that does not pass through the origin.

# %%
# TODO: Implement PerceptronNoBias by setting bias=0 and not updating it.
# This experiment is left as a guided exercise in the challenge workbook.
# Implementing it here reveals too much before the engineering challenge.
print("Experiment 4 (Remove Bias) is implemented as part of the engineering challenge.")
print("Attempt it in 07-engineering-challenge.md before running the solution here.")

# %% [markdown]
# ---
# ## Experiment 5: Learning Rate Comparison
#
# **Hypothesis:** Very high learning rates cause oscillation; very low ones cause
# slow convergence. A moderate rate (0.1) balances both.

# %%
import time

learning_rates = [0.001, 0.01, 0.1, 1.0, 10.0]
results_lr = []

for lr in learning_rates:
    p = PerceptronNumPy(learning_rate=lr, epochs=100)
    start = time.time()
    p.fit(OR_X, OR_y)
    elapsed = time.time() - start
    acc = p.score(OR_X, OR_y)
    results_lr.append((lr, acc, elapsed, p.weights.tolist(), p.bias))
    print(f"lr={lr:6.3f}: accuracy={acc:.0%}, weights={[f'{w:.3f}' for w in p.weights]}, bias={p.bias:.3f}")

# %% [markdown]
# **Observation:** _[fill in after running]_
#
# **Explanation:** _[fill in]_
#
# **Engineering implication:** _[fill in]_
#
# **Unexpected result:** _[fill in]_

# %% [markdown]
# ---
# ## Experiment 6: Training Order Shuffle
#
# **Hypothesis:** Shuffling the order in which training examples are presented
# may affect convergence speed but not the final result (for linearly separable data).

# %%
import random

random.seed(42)
shuffled_X = OR_X[:]
shuffled_y = OR_y[:]

combined = list(zip(shuffled_X, shuffled_y))
random.shuffle(combined)
shuffled_X, shuffled_y = zip(*combined)

p_shuffled = PerceptronNumPy(learning_rate=0.1, epochs=100)
p_shuffled.fit(list(shuffled_X), list(shuffled_y))
print(f"Shuffled order accuracy: {p_shuffled.score(OR_X, OR_y):.0%}")

p_ordered = PerceptronNumPy(learning_rate=0.1, epochs=100)
p_ordered.fit(OR_X, OR_y)
print(f"Original order accuracy: {p_ordered.score(OR_X, OR_y):.0%}")

print(f"\nWeights (shuffled): {p_shuffled.weights}")
print(f"Weights (ordered):  {p_ordered.weights}")
print(f"Same final weights: {np.allclose(p_shuffled.weights, p_ordered.weights)}")

# %% [markdown]
# **Observation:** _[fill in]_
#
# **Engineering implication:** Training order matters for the weight path but
# not necessarily for the final accuracy on linearly separable data.
# For non-separable data, shuffling may produce different local behaviors.

# %% [markdown]
# ---
# ## Experiment 7: Noisy Data

# %%
import random

random.seed(0)

def make_noisy_dataset(n=100, noise_rate=0.1, seed=0):
    """Generate a linearly separable dataset with random label noise."""
    random.seed(seed)
    np.random.seed(seed)
    X = np.random.rand(n, 2).tolist()
    y = [1 if xi[0] + xi[1] > 1.0 else 0 for xi in X]
    # Flip some labels randomly
    noisy_y = [1 - yi if random.random() < noise_rate else yi for yi in y]
    return X, noisy_y

X_noisy, y_noisy = make_noisy_dataset(n=200, noise_rate=0.1, seed=42)
p_noisy = PerceptronNumPy(learning_rate=0.1, epochs=100)
p_noisy.fit(X_noisy, y_noisy)
print(f"Noisy data accuracy: {p_noisy.score(X_noisy, y_noisy):.1%}")

# %% [markdown]
# **Observation:** _[fill in]_
#
# **Engineering implication:** The perceptron has no mechanism to handle noisy
# labels. It will try to classify every example correctly, including mislabeled ones,
# which can degrade the decision boundary. Logistic regression with regularization
# handles this more gracefully.

# %% [markdown]
# ---
# ## Experiment 8: Implementation Scaling (Plain Python vs NumPy)
#
# **Purpose:** Not to prove NumPy is always faster.
# To observe *when* vectorization becomes beneficial.
# Record: wall-clock time, lines of implementation code, readability (1–5),
# scaling behaviour.

# %%
import time

def make_linearly_separable(n, seed=0):
    """Generate n linearly separable samples."""
    np.random.seed(seed)
    X = np.random.rand(n, 2).tolist()
    y = [1 if xi[0] + xi[1] > 1.0 else 0 for xi in X]
    return X, y

sizes = [10, 1_000, 10_000]
results_scaling = []

for n in sizes:
    X, y = make_linearly_separable(n, seed=42)

    # Plain Python
    t0 = time.perf_counter()
    pp = PerceptronPython(learning_rate=0.1, epochs=10)
    pp.fit(X, y)
    t_python = time.perf_counter() - t0

    # NumPy
    t0 = time.perf_counter()
    pn = PerceptronNumPy(learning_rate=0.1, epochs=10)
    pn.fit(X, y)
    t_numpy = time.perf_counter() - t0

    ratio = t_python / t_numpy if t_numpy > 0 else float("inf")
    results_scaling.append((n, t_python, t_numpy, ratio))
    print(f"n={n:6d}: Python={t_python:.4f}s  NumPy={t_numpy:.4f}s  ratio={ratio:.1f}x")

# %% [markdown]
# **Observation:** _[fill in — is NumPy always faster? What is the crossover?]_
#
# **Explanation:** At small n, NumPy's overhead (array allocation, function call)
# may outweigh the benefit. At large n, vectorization dominates.
#
# **Engineering implication:** NumPy wins at scale, not necessarily at small problems.
# The choice between Python loops and NumPy is an engineering decision, not a rule.
#
# **Readability assessment:**
# - Plain Python: _/5 — [reason]_
# - NumPy:        _/5 — [reason]_
#
# **Unexpected result:** _[fill in]_
#
# **Reproducibility:**
# - Python version: (run `python --version`)
# - NumPy version: (run `numpy.__version__`)
# - Seed: 42
# - Epochs: 10
# - Dataset: random linearly separable (x₀ + x₁ > 1 → class 1)

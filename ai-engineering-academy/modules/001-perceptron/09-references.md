# 09 — References

## Primary Source

### Microsoft AI For Beginners — Perceptron Lesson

| Field | Value |
|---|---|
| **Title** | Introduction to Neural Networks: Perceptron |
| **Repository** | https://github.com/microsoft/AI-For-Beginners |
| **Path** | `lessons/3-NeuralNetworks/03-Perceptron/README.md` |
| **Notebook** | `lessons/3-NeuralNetworks/03-Perceptron/Perceptron.ipynb` |
| **Licence** | MIT |
| **Use** | Concept seed — module structure, experiments, and assessment are original |

---

## Authoritative Sources

### Rosenblatt, F. (1958)

**"The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"**  
Psychological Review, 65(6), 386–408.

The original paper. Defines the perceptron model, the learning rule, and the
convergence proof for linearly separable data.

Available via: Google Scholar, IEEE Xplore (may require institutional access).

---

### Minsky, M., & Papert, S. (1969)

**Perceptrons: An Introduction to Computational Geometry**  
MIT Press.

Proved the fundamental limitation of single-layer perceptrons — including the
inability to solve XOR — and catalyzed the first AI winter.

Historically important: understand that this book was about single-layer networks,
not multilayer ones. Multilayer networks were not widely explored until the 1980s.

---

## Pedagogical References

*(Used to compare teaching approaches — not copied into this module.)*

### Python Documentation — Built-in Types and Functions

https://docs.python.org/3.12/

Used for: understanding list iteration, `zip`, `enumerate`, and class design.

---

### NumPy Documentation — Array Operations

https://numpy.org/doc/stable/

Used for: `np.dot`, `np.zeros`, array broadcasting, `astype`.

---

### pytest Documentation

https://docs.pytest.org/en/stable/

Used for: writing the test contract in `tests/test_perceptron.py`.

---

## Further Reading

### Understanding linear separability

If you want an intuitive geometric explanation with visualizations:

**3Blue1Brown — Neural Networks series**  
https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

Episode 1 is about what neural networks are and how they compute. The perceptron
is the simplest case of what Brown explains.

---

### From perceptron to logistic regression

**Stanford CS229 Lecture Notes — Chapter 1**  
https://cs229.stanford.edu/notes2022fall/main_notes.pdf

Covers linear classifiers, the logistic regression extension of the perceptron,
and gradient descent — directly relevant to Modules 002 and 003.

---

### Where to find the original notebook

The Microsoft AI For Beginners Perceptron notebook demonstrates the algorithm on
the MNIST dataset, showing a progression beyond simple AND/OR gates.

After completing this module, that notebook is a useful next experiment:
`lessons/3-NeuralNetworks/03-Perceptron/Perceptron.ipynb`

It is in the cloned repository at the parent workspace level:
`../../../lessons/3-NeuralNetworks/03-Perceptron/Perceptron.ipynb`

---

## What to read before Module 002

Module 002 covers **loss functions**. Before starting it, these foundations help:

- Understand what a function minimum means geometrically.
- Be comfortable reading summation notation: $\sum_{i=1}^{n}$.
- Know the difference between a prediction and a probability.

No additional reading is required — Module 002 will build from the perceptron's update rule.

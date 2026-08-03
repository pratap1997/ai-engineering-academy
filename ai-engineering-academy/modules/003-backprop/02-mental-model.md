# Module 003: Mental Model — The Chain Rule on Computational DAGs

## 1. The Computational Graph as a Assembly Line

Think of a neural network as a multi-stage factory assembly line:
- **Inputs** ($x, y$) arrive at the start.
- **Workers** (Operations: `+`, `*`, `exp`, `tanh`) combine inputs into intermediate parts ($z_1, z_2, a_1$).
- **Final Inspection** computes the Error / Loss ($L$).

Forward propagation moves left to right, building the final product.

**Backpropagation moves right to left**, starting at the Loss ($L$). The Loss Inspector asks:
- "If I tweak intermediate part $a_1$ by $\epsilon$, how much does Loss $L$ change?" $\rightarrow \frac{\partial L}{\partial a_1}$
- The worker at $a_1$ then asks the worker at $z_1$: "Since my output affects Loss by $\frac{\partial L}{\partial a_1}$, and your output affects me by $\frac{\partial a_1}{\partial z_1}$, your total influence on Loss is $\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1}$."

This local product rule is the **Chain Rule of Calculus**.

---

## 2. Gradient Accumulation (The Multivariate Chain Rule)

What happens when a single variable $x$ is used in **multiple** downstream calculations?

For example:
$$y = x + x$$

Or a hidden node $h_1$ that feeds into two output nodes $y_1$ and $y_2$.

> 💡 **Multivariate Chain Rule**: If variable $x$ branches out to influence loss $L$ through multiple paths $p_1, p_2, \dots, p_k$, the total gradient of $L$ with respect to $x$ is the **SUM** of gradients along all paths:
> $$\frac{\partial L}{\partial x} = \sum_{i=1}^k \frac{\partial L}{\partial p_i} \frac{\partial p_i}{\partial x}$$

In code: We must **accumulate (`+=`)** gradients during backward traversal, not overwrite (`=`) them!

---

## 3. Reverse-Mode Autodiff in 3 Steps

1. **Forward Pass**: Build the Directed Acyclic Graph (DAG) by tracking parent-child relationships as operations execute.
2. **Topological Sort**: Order nodes in the DAG such that every node appears *after* all nodes that depend on it.
3. **Backward Pass**: Set $L.\text{grad} = 1.0$ and iterate backward through the topologically sorted list, calling each node's local `_backward()` closure to pass gradients to its parents.

# Module 003: Assessment & Readiness Check

## 1. Readiness Check (Formative Questions)

Review the following questions to verify your understanding of backpropagation & automatic differentiation:

### Q1: Gradient Accumulation
**Question**: Why does the `_backward()` implementation of binary operations (e.g. `+` or `*`) use `self.grad += ...` instead of `self.grad = ...`?
- **Answer**: When a variable is used multiple times in a computation graph (e.g. $y = x + x$), its total influence on the loss is the sum of gradients along all paths (multivariate chain rule). Overwriting with `=` would discard gradients from previous paths.

### Q2: Topological Sorting
**Question**: Why must backpropagation traverse nodes in reverse topological order?
- **Answer**: A node's total gradient `node.grad` must be fully accumulated from all downstream dependents *before* it propagates its gradient upstream to its parents. Reverse topological order guarantees all dependent children are processed first.

### Q3: Relative Error vs Absolute Error
**Question**: Why do we use relative error $\frac{|\text{analytical} - \text{numerical}|}{\max(|\text{analytical}|, |\text{numerical}|)}$ for `gradcheck` rather than raw absolute difference?
- **Answer**: Scale invariance. An absolute difference of $10^{-4}$ is huge if the gradient is $10^{-6}$, but negligible if the gradient is $10^4$. Relative error measures relative precision regardless of magnitude.

---

## 2. Capability Evaluation Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands that backpropagation uses the Chain Rule, but cannot explain topological sorting or gradient accumulation. |
| **Competent** | Can build a scalar `Value` node that correctly backpropagates through simple mathematical operations ($+$, $*$, $\text{relu}$, $\text{exp}$). |
| **Master** | Can derive matrix backprop equations, write custom tensor/batch loss nodes, and pass `gradcheck` tests ($< 10^{-6}$ error). |

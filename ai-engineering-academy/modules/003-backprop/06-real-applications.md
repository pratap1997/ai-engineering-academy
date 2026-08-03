# Module 003: Real Applications — Production Autodiff Frameworks

## 1. How Industrial Autodiff Frameworks Work

The `Value` scalar engine you built in `04-implementation.py` is the exact architectural foundation of modern AI frameworks:

| Framework | Autodiff Paradigm | Underlying Graph Mechanism |
|---|---|---|
| **PyTorch (`torch.autograd`)** | Dynamic (Tape-based) Reverse-Mode | Builds C++ DAG during forward execution; runs `tensor.backward()`. |
| **JAX (`jax.grad`)** | Functional Pure-Function Tracing | Traces Python functions into XLA HLO graphs; transforms code into derivative functions. |
| **TensorFlow (`tf.GradientTape`)** | Tape-recorded Context | Records operations inside `with tf.GradientTape() as tape:`. |
| **Micrograd** | Scalar Reverse-Mode | Andrej Karpathy's minimal educational engine (basis for our `Value` node). |

---

## 2. Symbolic vs Numerical vs Automatic Differentiation

Engineers often confuse the three methods of calculating derivatives:

1. **Numerical Differentiation (Finite Difference)**:
   $$\frac{\partial L}{\partial x} \approx \frac{L(x+\epsilon) - L(x-\epsilon)}{2\epsilon}$$
   - **Pros**: Easy to implement.
   - **Cons**: Slow ($O(N)$ evaluations for $N$ parameters), prone to floating-point truncation error.
   - **Best Use**: Gradient checking (`gradcheck`) only.

2. **Symbolic Differentiation (e.g., SymPy, Mathematica)**:
   Takes expression $f(x) = x^2 \sin(x)$ and outputs exact math string $f'(x) = 2x\sin(x) + x^2\cos(x)$.
   - **Pros**: Exact symbolic closed-form expressions.
   - **Cons**: Expression explosion problem for deep loops and control flows.

3. **Automatic Differentiation (Autodiff / Reverse-Mode)**:
   Decomposes any program into primitive mathematical steps (`+`, `*`, `sin`, `exp`) and applies the Chain Rule step-by-step.
   - **Pros**: Exact gradients, efficient memory management, handles arbitrary Python control flow (if statements, loops).
   - **Speed**: $O(1)$ time relative to forward pass runtime regardless of parameter count!

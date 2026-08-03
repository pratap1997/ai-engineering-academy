# Module 003: Engineering Challenge — Custom Computational Graph Operation

## 1. Challenge Context

In real-world machine learning systems, implementing every scalar operation individually in Python loops is too slow. Specialized loss functions like **Softmax Cross-Entropy** are combined into a single numerically stable C++/Python computational node with a closed-form analytical gradient.

Recall the Softmax Loss for a 2-class problem:
$$p_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2}}, \quad p_2 = \frac{e^{z_2}}{e^{z_1} + e^{z_2}}$$
$$\text{Loss} = -y_1 \log(p_1) - y_2 \log(p_2)$$

When you take the derivative of Softmax Cross-Entropy with respect to input $z_i$, the complicated quotient and log derivatives simplify to an remarkably elegant form:
$$\frac{\partial \text{Loss}}{\partial z_i} = p_i - y_i$$

---

## 2. Challenge Task

Implement a custom `SoftmaxCrossEntropyValue` class or method that extends our autodiff computational graph to handle 2-class logits $[z_1, z_2]$ and target one-hot labels $[y_1, y_2]$.

### Requirements:
1. **Numerically Stable Forward Pass**: Subtract $\max(z_1, z_2)$ before computing exponentiation to prevent overflow (`np.exp` overflow protection).
2. **Backward Pass**: Implement the analytical gradient $\frac{\partial L}{\partial z_i} = p_i - y_i$.
3. **Gradcheck Verification**: Verify your custom node against finite-difference numerical gradients using `eps=1e-5` to confirm relative error $< 10^{-6}$.

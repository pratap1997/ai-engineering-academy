# Module 003: Mathematics — Matrix Calculus & Backpropagation Derivations

## 1. Matrix Backpropagation Equations

Consider a 2-layer MLP with Mean Squared Error (MSE) loss:

$$\mathbf{z}^{(1)} = \mathbf{x} \mathbf{W}^{(1)T} + \mathbf{b}^{(1)}$$
$$\mathbf{a}^{(1)} = g^{(1)}\left(\mathbf{z}^{(1)}\right)$$
$$\mathbf{z}^{(2)} = \mathbf{a}^{(1)} \mathbf{W}^{(2)T} + \mathbf{b}^{(2)}$$
$$\mathbf{\hat{y}} = \mathbf{a}^{(2)} = g^{(2)}\left(\mathbf{z}^{(2)}\right)$$
$$L = \frac{1}{2} \|\mathbf{\hat{y}} - \mathbf{y}\|^2 = \frac{1}{2} \sum_{k} (\hat{y}_k - y_k)^2$$

### Step 1: Output Layer Pre-activation Gradient ($\boldsymbol{\delta}^{(2)}$)
$$\boldsymbol{\delta}^{(2)} = \frac{\partial L}{\partial \mathbf{z}^{(2)}} = \frac{\partial L}{\partial \mathbf{\hat{y}}} \odot g^{(2)\prime}\left(\mathbf{z}^{(2)}\right) = (\mathbf{\hat{y}} - \mathbf{y}) \odot g^{(2)\prime}\left(\mathbf{z}^{(2)}\right)$$

Where $\odot$ represents element-wise (Hadamard) multiplication.

### Step 2: Layer 2 Parameter Gradients
$$\frac{\partial L}{\partial \mathbf{W}^{(2)}} = \boldsymbol{\delta}^{(2)T} \mathbf{a}^{(1)}$$
$$\frac{\partial L}{\partial \mathbf{b}^{(2)}} = \boldsymbol{\delta}^{(2)}$$

### Step 3: Hidden Layer Pre-activation Gradient ($\boldsymbol{\delta}^{(1)}$)
$$\boldsymbol{\delta}^{(1)} = \frac{\partial L}{\partial \mathbf{z}^{(1)}} = \left(\boldsymbol{\delta}^{(2)} \mathbf{W}^{(2)}\right) \odot g^{(1)\prime}\left(\mathbf{z}^{(1)}\right)$$

### Step 4: Layer 1 Parameter Gradients
$$\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \boldsymbol{\delta}^{(1)T} \mathbf{x}$$
$$\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)}$$

---

## 2. Finite-Difference Numerical Gradient Checking (`gradcheck`)

To guarantee our analytical backpropagation formulas are 100% mathematically correct, we compare them against the finite-difference approximation:

$$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta_i + \epsilon) - L(\theta_i - \epsilon)}{2\epsilon}$$

Relative Error Formula:
$$\text{Relative Error} = \frac{|\text{grad}_\text{analytical} - \text{grad}_\text{numerical}|}{\max(|\text{grad}_\text{analytical}|, |\text{grad}_\text{numerical}|) + 10^{-8}}$$

- Relative Error $< 10^{-6} \rightarrow$ Analytical gradients are **CORRECT**.
- Relative Error $> 10^{-3} \rightarrow$ There is a bug in the backpropagation equations!

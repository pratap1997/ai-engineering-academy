# Module 002: Mathematics — Matrix Notation & Activation Functions

## 1. Multi-Layer Feedforward Matrix Notation

Let $L$ denote the total number of layers (including input layer $l=0$, hidden layers $l=1 \dots L-1$, and output layer $l=L$).

For layer $l$:
- $\mathbf{W}^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ is the weight matrix connecting layer $l-1$ to layer $l$.
- $\mathbf{b}^{(l)} \in \mathbb{R}^{n_l}$ is the bias vector for layer $l$.
- $\mathbf{z}^{(l)} \in \mathbb{R}^{n_l}$ is the pre-activation linear combination.
- $\mathbf{a}^{(l)} \in \mathbb{R}^{n_l}$ is the post-activation output vector ($\mathbf{a}^{(0)} = \mathbf{x}$).

The forward propagation equations for layer $l$ are:
$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$
$$\mathbf{a}^{(l)} = g^{(l)}\left(\mathbf{z}^{(l)}\right)$$

Where $g^{(l)}$ is the element-wise activation function of layer $l$.

---

## 2. Activation Functions

An activation function $g(z)$ must be non-linear to prevent layer collapse.

### 2.1 Heaviside Step Function
$$g(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases}$$
- **Range**: $\{0, 1\}$
- **Derivative**: $0$ everywhere except $z=0$ (undefined/infinite spike). Cannot be used with gradient-based learning (backpropagation).

### 2.2 Sigmoid (Logistic) Function
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Range**: $(0, 1)$
- **Derivative**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- **Properties**: Smooth, probabilistic interpretation, but suffers from vanishing gradients for $|z| \gg 0$.

### 2.3 Hyperbolic Tangent (Tanh)
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1$$
- **Range**: $(-1, 1)$
- **Derivative**: $\tanh'(z) = 1 - \tanh^2(z)$
- **Properties**: Zero-centered, generally trains faster than Sigmoid.

### 2.4 Rectified Linear Unit (ReLU)
$$\text{ReLU}(z) = \max(0, z)$$
- **Range**: $[0, \infty)$
- **Derivative**: $\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}$
- **Properties**: Computationally fast, no vanishing gradient for positive values. Standard choice in modern deep learning.

---

## 3. Vectorized Batch Forward Propagation

When processing $N$ samples simultaneously in a batch matrix $\mathbf{X} \in \mathbb{R}^{N \times n_0}$ (where each row is one sample $\mathbf{x}_i^T$):

$$\mathbf{Z}^{(1)} = \mathbf{X} \mathbf{W}^{(1)T} + \mathbf{b}^{(1)T}$$
$$\mathbf{A}^{(1)} = g^{(1)}\left(\mathbf{Z}^{(1)}\right)$$
$$\mathbf{Z}^{(2)} = \mathbf{A}^{(1)} \mathbf{W}^{(2)T} + \mathbf{b}^{(2)T}$$
$$\mathbf{A}^{(2)} = g^{(2)}\left(\mathbf{Z}^{(2)}\right)$$

Where:
- $\mathbf{X} \in \mathbb{R}^{N \times n_0}$
- $\mathbf{W}^{(1)} \in \mathbb{R}^{n_1 \times n_0}$ (or transposed to $\mathbb{R}^{n_0 \times n_1}$)
- $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times n_1}$ (broadcast across $N$ rows)
- $\mathbf{A}^{(2)} \in \mathbb{R}^{N \times n_2}$ (final model prediction batch)

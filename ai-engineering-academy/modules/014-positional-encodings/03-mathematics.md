# Module 014: Mathematics — RoPE, ALiBi & Relative Position Formulations

## 1. Rotary Position Embedding (RoPE) Mathematical Derivation

For a 2D vector $\mathbf{x} = (x_1, x_2)^T$, rotation by angle $m\theta_i$ is defined by matrix:

$$\mathbf{R}_{\Theta, m}^{(i)} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

For a $d$-dimensional vector, partition into $d/2$ pairs of 2D coordinates:

$$\mathbf{R}_{\Theta, m} = \text{diag}\left(\mathbf{R}_{\Theta, m}^{(1)}, \mathbf{R}_{\Theta, m}^{(2)}, \dots, \mathbf{R}_{\Theta, m}^{(d/2)}\right)$$

Where frequencies $\theta_i$ follow the geometric progression:
$$\theta_i = 10000^{-2(i-1)/d}, \quad i \in \{1, 2, \dots, d/2\}$$

### The Inner Product Property
For query $\mathbf{q}$ at position $m$ and key $\mathbf{k}$ at position $n$:

$$\langle \mathbf{R}_{\Theta, m} \mathbf{q}, \mathbf{R}_{\Theta, n} \mathbf{k} \rangle = \mathbf{q}^T \mathbf{R}_{\Theta, m}^T \mathbf{R}_{\Theta, n} \mathbf{k} = \mathbf{q}^T \mathbf{R}_{\Theta, n - m} \mathbf{k}$$

Since $\mathbf{R}_{\Theta, m}^T \mathbf{R}_{\Theta, n} = \mathbf{R}_{\Theta, n - m}$, the dot product depends solely on the relative displacement $(n - m)$.

### Efficient Computation
Instead of full matrix multiplication, RoPE can be computed via elementwise ops:

$$\mathbf{R}_{\Theta, m} \mathbf{x} = \mathbf{x} \odot \cos(m\Theta) + \tilde{\mathbf{x}} \odot \sin(m\Theta)$$

Where $\tilde{\mathbf{x}} = (-x_2, x_1, -x_4, x_3, \dots, -x_d, x_{d-1})^T$.

---

## 2. ALiBi Slopes & Distance Penalty

Given sequence length $T$ and head index $h \in \{1, \dots, H\}$:

$$S_{h, i, j} = \frac{\mathbf{q}_i \mathbf{k}_j^T}{\sqrt{d_k}} - m_h \cdot |i - j|$$

Where head slopes $m_h$ are geometric ratios:
For $H = 8$:
$$m_h \in \left\{ \frac{1}{2^1}, \frac{1}{2^2}, \frac{1}{2^3}, \dots, \frac{1}{2^8} \right\} = \left\{ \frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}, \frac{1}{64}, \frac{1}{128}, \frac{1}{256} \right\}$$

---

## 3. T5 Logarithmic Relative Bucketing

For position $i$ and $j$, relative distance $d = i - j$:

$$\text{bucket}(d) = \begin{cases}
d & \text{if } |d| < 8 \\
8 + \left\lfloor \frac{\log(|d| / 8)}{\log(\text{max\_distance} / 8)} \times (\text{num\_buckets} - 8) \right\rfloor & \text{if } |d| \ge 8
\end{cases}$$

The bucket index looks up a scalar bias added to attention score $S_{i, j}$.

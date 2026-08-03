# Module 006: Mental Model — Sliding Filters & Feature Hierarchies

## 1. The Sliding Window Analogy

Imagine holding a magnifying glass ($3 \times 3$ grid) over a large photograph:
- At each position, you multiply the grid values by the pixels under the glass, sum them up, and write the single resulting number onto a new sheet of paper (the **Feature Map**).
- If the filter is designed to detect **vertical edges**, the feature map lights up with high values wherever a sharp vertical boundary exists in the image.

---

## 2. Classic Hand-Crafted Kernels vs Learnable Kernels

Before deep learning, computer vision engineers hand-crafted filter matrices:

### Vertical Sobel Edge Filter ($K = 3 \times 3$)
$$\mathbf{K}_\text{vertical} = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$
When slid across a image, pixels with uniform intensity cancel out to $0.0$, while left-to-right intensity transitions produce large positive or negative activation values.

### Horizontal Sobel Edge Filter ($K = 3 \times 3$)
$$\mathbf{K}_\text{horizontal} = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

In **CNNs**, we do **not** hand-craft these values. We initialize random weights for $K$ and use **Backpropagation** to automatically learn optimal filters for the task!

---

## 3. Max Pooling — Local Invariance & Downsampling

After extracting spatial features with Conv2D, **Max Pooling** downsamples spatial dimensions by taking the maximum value within small $2 \times 2$ non-overlapping patches:

$$\begin{bmatrix} 1 & 3 \\ 2 & 9 \end{bmatrix} \xrightarrow{\text{MaxPool } 2 \times 2} \begin{bmatrix} 9 \end{bmatrix}$$

- **Spatial Invariance**: If a feature (e.g. an eye) shifts by $1$ pixel, the maximum within the $2 \times 2$ patch remains unchanged!
- **Dimensionality Reduction**: Reduces spatial area by $75\%$ ($2 \times 2 \rightarrow 1 \times 1$), cutting memory and computation for downstream layers.

# Module 011: Mathematics — FFN, LayerNorm & Residual Gradient Flow

## 1. Position-wise Feed-Forward Network (FFN)

Applied **identically and independently** to each token position $t$:

$$\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$

Where:
- $\mathbf{W}_1 \in \mathbb{R}^{d_\text{model} \times d_{ff}}$, $\mathbf{b}_1 \in \mathbb{R}^{d_{ff}}$ — expansion projection
- $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_\text{model}}$, $\mathbf{b}_2 \in \mathbb{R}^{d_\text{model}}$ — compression projection
- $d_{ff} = 4 \times d_\text{model}$ (standard; 768→3072 in BERT-base, 512→2048 in the original paper)

**GELU** (Gaussian Error Linear Unit):
$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)\right)\right)$$

---

## 2. Layer Normalization

For a token vector $\mathbf{x} \in \mathbb{R}^{d_\text{model}}$:

$$\mu = \frac{1}{d} \sum_{j=1}^d x_j, \quad \sigma^2 = \frac{1}{d} \sum_{j=1}^d (x_j - \mu)^2$$
$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where $\gamma, \beta \in \mathbb{R}^{d_\text{model}}$ are learned scale and shift parameters initialized to $\mathbf{1}$ and $\mathbf{0}$.

**Key difference from BatchNorm**: LayerNorm normalizes across the **feature dimension** per sample, making it sequence-length and batch-size independent.

---

## 3. Residual Gradient Flow

For any sublayer $F$ (MHA or FFN), the Pre-LN residual connection:

$$\mathbf{y} = \mathbf{x} + F(\text{LN}(\mathbf{x}))$$

The gradient with respect to $\mathbf{x}$:
$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{I} + \frac{\partial F(\text{LN}(\mathbf{x}))}{\partial \mathbf{x}}$$

The **identity term $\mathbf{I}$** ensures that gradients flow directly from output to input regardless of what happens in $F(\cdot)$ — identical to the ResNet skip connection from Module 007.

---

## 4. Full Transformer Encoder Block (Pre-LN)

$$\mathbf{x}' = \mathbf{x} + \text{MHA}(\text{LN}_1(\mathbf{x}), \text{LN}_1(\mathbf{x}), \text{LN}_1(\mathbf{x}))$$
$$\mathbf{x}'' = \mathbf{x}' + \text{FFN}(\text{LN}_2(\mathbf{x}'))$$

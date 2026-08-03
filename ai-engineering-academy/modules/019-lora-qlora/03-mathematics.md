# Module 019: Mathematics — Low-Rank Decomposition & QLoRA Scaling

## 1. LoRA Forward Pass & Weight Update Equations

Given input $\mathbf{x} \in \mathbb{R}^{B \times d_\text{in}}$, base weight $\mathbf{W}_0 \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$:

Trainable adapter matrices:
$$\mathbf{A} \in \mathbb{R}^{r \times d_\text{in}}, \quad \mathbf{B} \in \mathbb{R}^{d_\text{out} \times r}$$

Initialization:
- $\mathbf{A} \sim \mathcal{N}\left(0, \frac{1}{r}\right)$ (Gaussian distribution)
- $\mathbf{B} = \mathbf{0}$ (Zero initialization)

This ensures that at step 0:
$$\Delta \mathbf{W} = \frac{\alpha}{r} \mathbf{B} \mathbf{A} = \mathbf{0}$$

Model behavior at $t=0$ is **exactly identical** to the pre-trained base model!

### Forward Pass Computation:
$$\mathbf{h} = \mathbf{x} \mathbf{W}_0^T + \frac{\alpha}{r} (\mathbf{x} \mathbf{A}^T) \mathbf{B}^T$$

Where $\alpha$ is a scaling hyperparameter (typically $\alpha = 2r$ or $\alpha = 16$).

---

## 2. Weight Merging & Unmerging

**Merge for Deployment**:
$$\mathbf{W}_{\text{merged}} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}$$

After merging, forward pass becomes standard matrix multiplication:
$$\mathbf{h} = \mathbf{x} \mathbf{W}_{\text{merged}}^T$$

**Unmerge**:
$$\mathbf{W}_0 = \mathbf{W}_{\text{merged}} - \frac{\alpha}{r} \mathbf{B} \mathbf{A}$$

---

## 3. QLoRA (Quantized LoRA) Integration

In QLoRA, base weight $\mathbf{W}_0$ is quantized into 4-bit representation ($Q(\mathbf{W}_0)$ with block scale $S$).

During forward pass:
$$\mathbf{W}_{\text{dequant}} = \text{Dequantize}(Q(\mathbf{W}_0), S)$$
$$\mathbf{h} = \mathbf{x} \mathbf{W}_{\text{dequant}}^T + \frac{\alpha}{r} (\mathbf{x} \mathbf{A}^T) \mathbf{B}^T$$

Base weight gradients $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_0}$ are **never computed**, avoiding FP32 optimizer memory overhead!

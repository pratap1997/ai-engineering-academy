# Module 019: Assessment & Readiness Check

## 1. Formative Questions

### Q1: Why does LoRA initialize Matrix $A$ with Gaussian noise and Matrix $B$ with zeroes?
**Answer**: Initializing $\mathbf{B} = \mathbf{0}$ ensures that at step 0 of training, the adapter update $\Delta \mathbf{W} = \frac{\alpha}{r} \mathbf{B} \mathbf{A} = \mathbf{0}$. Therefore, the initial forward pass of the model is exactly identical to the pre-trained base model, avoiding initial loss spikes.

### Q2: How does LoRA eliminate latency penalty during inference serving?
**Answer**: Prior to deployment, the low-rank update $\Delta \mathbf{W} = \frac{\alpha}{r} \mathbf{B} \mathbf{A}$ is added directly to base weight $\mathbf{W}_0$ ($\mathbf{W}_\text{merged} = \mathbf{W}_0 + \Delta \mathbf{W}$). During serving, execution uses the single merged weight matrix, eliminating all low-rank matrix multiplications.

### Q3: What is the core innovation of QLoRA over standard LoRA?
**Answer**: QLoRA quantizes frozen base model weights $\mathbf{W}_0$ into 4-bit NormalFloat4 (NF4) representation with double quantization, while keeping trainable LoRA adapters in 16-bit/32-bit floating point. This enables fine-tuning a 70B model on a single 48GB GPU.

---

## 2. Capability Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands why parameter-efficient fine-tuning saves optimizer VRAM |
| **Competent** | Can implement `LoRALinear` with scaling factor $\alpha / r$ and weight merging |
| **Master** | Can implement `QLoRALinear`, build a multi-adapter dynamic switching layer, and calculate exact parameter reduction ratios |

# Module 005: Assessment & Readiness Check

## 1. Readiness Check (Formative Questions)

Review the following questions to verify your operational understanding of regularization & normalization:

### Q1: L1 vs L2 Weight Penalties
**Question**: Why does L1 regularization drive weights to exact zeros (sparsity), whereas L2 only shrinks them smoothly?
- **Answer**: The L1 constraint boundary is a diamond with sharp corners located on the axes where parameters equal zero. Loss contours expanding from the center hit these axial corners with high probability. L2 boundaries are smooth spheres, so contours hit arbitrary non-zero points.

### Q2: Inverted Dropout Scaling
**Question**: Why do we divide surviving activations by $(1 - p)$ during training in Inverted Dropout?
- **Answer**: Dividing by $(1 - p)$ during training ensures that the expected value of activations remains identical between training and test modes ($\mathbb{E}[A_\text{train}] = \mathbb{E}[A_\text{eval}] = A$). This avoids needing to scale weights by $(1 - p)$ at test time.

### Q3: BatchNorm vs LayerNorm
**Question**: Why is LayerNorm preferred over BatchNorm in Transformer networks and Large Language Models?
- **Answer**: BatchNorm computes statistics across the mini-batch dimension $N$. In NLP and Transformers, batch sizes vary dynamically, sequence lengths differ, and sequence order matters. LayerNorm normalizes across the feature dimension $D$ independently for each token/sample, making it invariant to batch size and sequence length.

---

## 2. Capability Evaluation Rubric

| Level | Criteria |
|---|---|
| **Novice** | Can define overfitting and explain why Dropout drops units randomly. |
| **Competent** | Can write code for Inverted Dropout and explain the difference between train and eval modes. |
| **Master** | Can derive BatchNorm and LayerNorm forward/backward passes, implement running mean/var tracking, and pass `gradcheck` tests ($< 10^{-5}$ error). |

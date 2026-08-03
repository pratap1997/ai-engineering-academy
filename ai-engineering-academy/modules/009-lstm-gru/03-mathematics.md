# Module 009: Mathematics — Gated Memory Equations & CEC Proof

## 1. Long Short-Term Memory (LSTM) Equations

For step input $\mathbf{x}_t \in \mathbb{R}^{N \times D}$ and previous hidden state $\mathbf{h}_{t-1} \in \mathbb{R}^{N \times H}$:

### 1.1 Gate Equations:
$$\mathbf{f}_t = \sigma(\mathbf{x}_t \mathbf{W}_{xf} + \mathbf{h}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f) \quad \text{(Forget Gate)}$$
$$\mathbf{i}_t = \sigma(\mathbf{x}_t \mathbf{W}_{xi} + \mathbf{h}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i) \quad \text{(Input Gate)}$$
$$\mathbf{\tilde{C}}_t = \tanh(\mathbf{x}_t \mathbf{W}_{xc} + \mathbf{h}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c) \quad \text{(Candidate Cell)}$$
$$\mathbf{o}_t = \sigma(\mathbf{x}_t \mathbf{W}_{xo} + \mathbf{h}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o) \quad \text{(Output Gate)}$$

### 1.2 State Updates:
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \mathbf{\tilde{C}}_t \quad \text{(Cell State)}$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t) \quad \text{(Hidden State)}$$

---

## 2. Mathematical Proof of Constant Error Carousel (CEC)

Why does the Cell State $\mathbf{C}_t$ eliminate vanishing gradients during BPTT?

Consider the gradient derivative of $\mathbf{C}_t$ with respect to $\mathbf{C}_{t-1}$:
$$\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = \mathbf{f}_t + \frac{\partial \mathbf{f}_t}{\partial \mathbf{C}_{t-1}} \mathbf{C}_{t-1} + \frac{\partial \mathbf{i}_t}{\partial \mathbf{C}_{t-1}} \mathbf{\tilde{C}}_t + \frac{\partial \mathbf{\tilde{C}}_t}{\partial \mathbf{C}_{t-1}} \mathbf{i}_t$$

If the network learns to set the Forget Gate $\mathbf{f}_t \approx 1.0$:
$$\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} \approx 1.0$$

The product of cell state derivatives across $T$ steps becomes:
$$\prod_{k=2}^T \frac{\partial \mathbf{C}_k}{\partial \mathbf{C}_{k-1}} \approx \mathbf{1.0}$$

**Conclusion**: The gradient flows backward across hundreds of time steps with zero exponential decay!

---

## 3. Gated Recurrent Unit (GRU) Equations

$$\mathbf{r}_t = \sigma(\mathbf{x}_t \mathbf{W}_{xr} + \mathbf{h}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r) \quad \text{(Reset Gate)}$$
$$\mathbf{z}_t = \sigma(\mathbf{x}_t \mathbf{W}_{xz} + \mathbf{h}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z) \quad \text{(Update Gate)}$$
$$\mathbf{\tilde{h}}_t = \tanh(\mathbf{x}_t \mathbf{W}_{xh} + (\mathbf{r}_t \odot \mathbf{h}_{t-1}) \mathbf{W}_{hh} + \mathbf{b}_h) \quad \text{(Candidate Hidden)}$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \mathbf{\tilde{h}}_t \quad \text{(Hidden State)}$$

---

## 4. Parameter Count Comparison

For hidden dimension $H$ and input dimension $D$:

- **Vanilla RNN**: $1 \times (H \cdot D + H \cdot H + H)$ parameters.
- **GRU**: $3 \times (H \cdot D + H \cdot H + H)$ parameters ($3\times$ Vanilla RNN).
- **LSTM**: $4 \times (H \cdot D + H \cdot H + H)$ parameters ($4\times$ Vanilla RNN).

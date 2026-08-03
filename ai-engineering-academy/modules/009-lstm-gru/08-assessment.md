# Module 009: Assessment & Readiness Check

## 1. Readiness Check (Formative Questions)

Review the following questions to verify your operational understanding of LSTMs & GRUs:

### Q1: The Constant Error Carousel (CEC)
**Question**: Why does the LSTM Cell State $\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \mathbf{\tilde{C}}_t$ solve the vanishing gradient problem?
- **Answer**: The Cell State update is **additive** rather than multiplicative. The derivative $\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = \mathbf{f}_t$. When the Forget Gate $\mathbf{f}_t \approx 1.0$, the gradient derivative is $1.0$, allowing gradients to flow backward through hundreds of steps without exponential decay.

### Q2: Forget Gate Bias Initialization
**Question**: Why is it standard practice to initialize the Forget Gate bias $\mathbf{b}_f$ to $+1.0$ or $+2.0$?
- **Answer**: Initializing $\mathbf{b}_f = +1.0$ forces $\sigma(\mathbf{b}_f) \approx 0.73 \dots 0.88$ at the start of training. This ensures that the network defaults to *remembering* previous information rather than accidentally erasing memory during early iterations.

### Q3: LSTM vs GRU Parameter Tradeoffs
**Question**: How do parameter counts compare between LSTM and GRU for hidden size $H$ and input dimension $D$?
- **Answer**: LSTM uses 4 weight matrices ($4 \times (H \cdot D + H^2 + H)$), whereas GRU uses 3 weight matrices ($3 \times (H \cdot D + H^2 + H)$). GRU provides a $25\%$ reduction in parameter count and runs faster with comparable performance.

---

## 2. Capability Evaluation Rubric

| Level | Criteria |
|---|---|
| **Novice** | Understands LSTM gates conceptually, but cannot write an `LSTMCell` forward pass. |
| **Competent** | Can implement `LSTMCell`, `GRUCell`, and unrolled sequences in Python. |
| **Master** | Can derive Constant Error Carousel (CEC) gradient proofs, explain GRU coupling, build sequence-to-sequence models, and pass `gradcheck` tests ($< 10^{-4}$ error). |

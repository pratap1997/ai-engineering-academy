# Module 008: Mental Model — Unrolling Time & Exploding Gradient Cliffs

## 1. Unrolling Recurrent Networks Across Time

Think of an RNN as a single physical node with a feedback loop:
- In the time domain, we **unroll** this loop into a chain of $T$ identical nodes sharing the **exact same weight matrices** ($\mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{W}_{hy}$).

```
h_0 ---> [ Cell 1 ] ---> h_1 ---> [ Cell 2 ] ---> h_2 ---> ... ---> [ Cell T ] ---> h_T
            ^                        ^                                 ^
            |                        |                                 |
           x_1                      x_2                               x_T
```

Every step $t$ takes the previous memory $\mathbf{h}_{t-1}$ and current input $\mathbf{x}_t$, updates the memory to $\mathbf{h}_t$, and emits an output $\mathbf{y}_t$.

---

## 2. Exploding & Vanishing Gradients in Time

During Backpropagation Through Time (BPTT), the gradient at step $T$ must travel all the way back to step $1$.
This requires repeatedly multiplying by the recurrent weight matrix transpose $\mathbf{W}_{hh}^T$:

$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_1} = \prod_{k=2}^T \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} \propto (\mathbf{W}_{hh}^T)^{T-1}$$

Imagine multiplying a number $w$ by itself 50 times ($w^{50}$):
- **If largest eigenvalue $\rho(\mathbf{W}_{hh}) > 1.0$**: Gradient magnitude explodes exponentially ($1.2^{50} \approx 9,100$), causing numerical `NaN` / `Inf` crashes!
- **If largest eigenvalue $\rho(\mathbf{W}_{hh}) < 1.0$**: Gradient vanishes exponentially ($0.8^{50} \approx 0.000014$), causing the network to forget long-term context.

---

## 3. Gradient Clipping — The Parachute Safety Net

To prevent exploding gradients from causing catastrophic parameter destruction, we apply **Gradient Clipping by Norm**:

$$\text{If } \|\mathbf{g}\| > M, \quad \mathbf{g} \leftarrow \mathbf{g} \cdot \frac{M}{\|\mathbf{g}\|}$$

- **Intuition**: If the gradient vector norm exceeds safety threshold $M$ (e.g. $1.0$), we scale down its magnitude while preserving its exact directional vector!
- This acts as an emergency brake / parachute whenever training hits a steep loss cliff.

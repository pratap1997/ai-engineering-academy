# Module 004: Mental Model — Loss Landscape Traversal

## 1. Physical Analogies for Optimizers

Think of minimizing a loss function $L(\mathbf{\theta})$ as navigating a hilly 3D landscape in total fog:

### 1. Vanilla SGD — The Careful Hiker
- **Mechanism**: Takes a step proportional to the steepness underfoot.
- **Flaw**: If the hiker enters a narrow canyon, they bounce back and forth between steep canyon walls rather than walking down the stream at the canyon floor.

### 2. Momentum — The Heavy Iron Ball
- **Mechanism**: Imagine rolling a heavy iron ball down the hill. As it rolls, it builds up momentum (velocity $v$).
- **Advantage**: The lateral oscillations across canyon walls cancel out, while forward momentum down the canyon builds up speed, smashing through small bumps and plateaus!

### 3. RMSprop — Adaptive Surface Friction
- **Mechanism**: Tracks the magnitude of recent gradients for *each* parameter separately. If a parameter has huge gradient spikes, RMSprop increases friction (divides step size by $\sqrt{v}$). If a parameter has tiny gradients, RMSprop decreases friction.
- **Advantage**: Prevents gradient explosion along steep dimensions while accelerating progress along gentle slopes.

### 4. Adam — The Jetpack (Momentum + RMSprop)
- **Mechanism**: Combines Momentum (1st moment $m_t$, directional velocity) AND RMSprop (2nd moment $v_t$, parameter-wise adaptive friction), plus **bias correction** for early iterations.
- **Advantage**: Fast, robust, and the default choice for modern deep learning.

---

## 2. Adam vs. AdamW: Decoupled Weight Decay

In classical SGD, L2 regularization (penalty $\frac{\lambda}{2} \|\theta\|^2$) is mathematically equivalent to weight decay:
$$\mathbf{g}_t = \nabla L(\theta_t) + \lambda \theta_t$$
$$\theta_{t+1} = \theta_t - \eta \mathbf{g}_t = (1 - \eta \lambda)\theta_t - \eta \nabla L(\theta_t)$$

However, in **Adam**, gradients are divided by $\sqrt{v_t}$:
$$\mathbf{g}_t^\text{regularized} = \nabla L(\theta_t) + \lambda \theta_t$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \left( \hat{m}_t^\text{regularized} \right)$$

This means weights with **large historical gradients** (large $v_t$) receive **LESS weight decay**, while weights with **small gradients** receive **MORE weight decay**! This distorts regularization.

> 💡 **AdamW Solution**: Decouple weight decay from the gradient moving averages:
> $$\theta_{t+1} = (1 - \eta \lambda)\theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
> AdamW applies weight decay directly to the parameters *after* computing the adaptive step.

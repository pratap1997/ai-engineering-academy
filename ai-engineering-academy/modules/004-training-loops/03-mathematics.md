# Module 004: Mathematics — Optimizer Update Equations & Schedulers

## 1. Optimizer Update Formulas

Let $g_t = \nabla_\theta L(\theta_t)$ be the gradient at step $t$.

### 1.1 SGD with Momentum
$$v_t = \beta v_{t-1} + g_t$$
$$\theta_{t+1} = \theta_t - \eta v_t$$
where $\beta \in [0, 1)$ is the momentum coefficient (typically $0.9$).

### 1.2 RMSprop
$$v_t = \alpha v_{t-1} + (1 - \alpha) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} g_t$$
where $\alpha$ is the smoothing constant (typically $0.99$) and $\epsilon = 10^{-8}$.

### 1.3 Adam (Adaptive Moment Estimation)
1. First moment (mean of gradients):
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
2. Second moment (uncentered variance of gradients):
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
3. Bias correction (corrects zero-initialization at early steps $t=1, 2, \dots$):
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
4. Parameter update:
   $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
Default hyperparameters: $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$.

### 1.4 AdamW (Decoupled Weight Decay)
$$\theta_{t+1} = (1 - \eta \lambda) \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
where $\lambda$ is the weight decay factor.

---

## 2. Learning Rate Schedulers

### 2.1 Step Decay Scheduler
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / S \rfloor}$$
where $\gamma < 1$ is decay rate and $S$ is step size in epochs.

### 2.2 Cosine Annealing Scheduler
$$\eta_t = \eta_\min + \frac{1}{2} (\eta_0 - \eta_\min) \left( 1 + \cos\left( \frac{t}{T_\max} \pi \right) \right)$$
where $T_\max$ is total steps/epochs and $\eta_\min$ is floor learning rate (typically $0.0$).

# 03 - Mathematics of Long-Context Scaling

## Standard RoPE
Rotary Position Embedding applies a rotation matrix to the query and key vectors.
For position $m$ and dimension index $i \in [1, d/2]$:
$$ \theta_i = 10000^{-2(i-1)/d} $$
The rotation is defined as:
$$ \mathbf{R}_{\Theta, m}^d = \text{diag}(R_{\theta_1, m}, \dots, R_{\theta_{d/2}, m}) $$
where $R_{\theta_i, m} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$.

## Linear Position Interpolation
To scale context by a factor $s = L_{new} / L_{train}$, we interpolate the position index $m$:
$$ m' = \frac{m}{s} $$
Or equivalently, modify the frequencies:
$$ \theta_i' = \frac{\theta_i}{s} $$

## NTK-Aware Scaling
Instead of linear scaling, NTK-aware scaling changes the base $B$ (default 10000):
$$ B' = B \cdot s^{\frac{d}{d-2}} $$
This ensures high frequencies are minimally affected while low frequencies are scaled aggressively.

## YaRN (Yet Another RoPE Extrapolation)
YaRN defines a ramp function based on the wavelength of the frequencies:
$$ \gamma_i = \begin{cases} 0 & \text{if } \lambda_i < \beta_{fast} \\ 1 & \text{if } \lambda_i > \beta_{slow} \\ \text{linear interpolation} & \text{otherwise} \end{cases} $$
The scaled frequency is:
$$ \theta_i' = \theta_i \cdot s^{-\gamma_i} $$
YaRN also multiplies the attention logits by a temperature $t = \sqrt{1 + 0.1 \ln(s)}$ to maintain entropy.

## Sliding Window Attention Mask
The attention mask $M$ for sequence length $N$ and window size $W$:
$$ M_{ij} = \begin{cases} 0 & \text{if } 0 \le i - j \le W \\ -\infty & \text{otherwise} \end{cases} $$

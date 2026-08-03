# Module 016: Mathematics — Quantization Formulas & Scale Computation

## 1. Asymmetric Quantization Equations

Given real-valued tensor $X \in \mathbb{R}$, target integer range $[q_{\min}, q_{\max}]$:

For unsigned INT8: $[q_{\min}, q_{\max}] = [0, 255]$.
For signed INT8: $[q_{\min}, q_{\max}] = [-128, 127]$.

**Step 1: Compute Scale ($s$)**:
$$s = \frac{\max(X) - \min(X)}{q_{\max} - q_{\min}}$$

**Step 2: Compute Zero-Point ($z$)**:
$$z = \text{round}\!\left(\frac{-\min(X)}{s}\right) + q_{\min}$$
$$z = \text{clamp}(z, q_{\min}, q_{\max})$$

**Step 3: Quantize**:
$$Q(X) = \text{clamp}\!\left(\text{round}\!\left(\frac{X}{s}\right) + z, q_{\min}, q_{\max}\right)$$

**Step 4: Dequantize**:
$$\hat{X} = s \cdot (Q(X) - z) \approx X$$

---

## 2. Symmetric Quantization Equations

For symmetric signed $b$-bit integer quantization ($q_{\max} = 2^{b-1} - 1$):
For INT8: $q_{\max} = 127$. For INT4: $q_{\max} = 7$.

**Step 1: Compute Scale ($s$)**:
$$\alpha = \max(|X|)$$
$$s = \frac{\alpha}{q_{\max}}$$

**Step 2: Quantize ($z=0$)**:
$$Q(X) = \text{clamp}\!\left(\text{round}\!\left(\frac{X}{s}\right), -q_{\max}, q_{\max}\right)$$

**Step 3: Dequantize**:
$$\hat{X} = s \cdot Q(X)$$

---

## 3. Mean Squared Quantization Error (MSE)

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (X_i - \hat{X}_i)^2$$

Quantization Signal-to-Noise Ratio (SNR):
$$\text{SQNR (dB)} = 10 \log_{10} \frac{\sum X_i^2}{\sum (X_i - \hat{X}_i)^2}$$

Every bit added to quantization accuracy improves SQNR by approximately $6.02\text{ dB}$.

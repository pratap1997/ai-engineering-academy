# 03 - Mathematics of AI Gateways

AI Gateways rely on deterministic rules and heuristics to optimize cost, latency, and reliability. The core mathematical models revolve around compression ratios, cache similarity, and routing cost functions.

## 1. Token Compression Ratio

Token compression algorithms aim to reduce the prompt length without losing semantic meaning. The effectiveness is measured by the compression ratio $C$.

$$C = 1 - \frac{|T_{compressed}|}{|T_{original}|}$$

Where:
- $|T_{original}|$ is the number of tokens in the raw prompt.
- $|T_{compressed}|$ is the number of tokens after applying compression heuristics (e.g., stop-word removal, RTK, Caveman rules).

A compression ratio of 0.20 means the prompt size was reduced by 20%, directly correlating to a 20% reduction in input token costs.

## 2. Semantic Cache Similarity Threshold

To implement a semantic cache, we embed the incoming query into a high-dimensional vector space and compare it against previously cached queries using cosine similarity.

$$\text{sim}(\mathbf{q}_{new}, \mathbf{q}_{cached}) = \frac{\mathbf{q}_{new} \cdot \mathbf{q}_{cached}}{\|\mathbf{q}_{new}\| \|\mathbf{q}_{cached}\|}$$

A cache hit occurs if the similarity exceeds a predefined threshold $\theta$:

$$\text{Hit} \iff \exists \, \mathbf{q}_{cached} \in \text{Cache} \text{ such that } \text{sim}(\mathbf{q}_{new}, \mathbf{q}_{cached}) \ge \theta$$

Choosing $\theta$ is a precision-recall tradeoff. 
- $\theta \approx 0.99$: Very strict, acts almost like an exact cache (high precision, low recall).
- $\theta \approx 0.85$: Loose, catches variations but risks serving irrelevant answers (lower precision, high recall).

## 3. Provider Priority Cost Function

When selecting a provider dynamically (Cost-based or Latency-based routing), the gateway evaluates a cost function $Cost(p)$ for each available provider $p$.

$$Cost(p) = w_1 \cdot \text{Latency}(p) + w_2 \cdot \text{Price}(p) + w_3 \cdot \text{ErrorRate}(p)$$

Where:
- $w_1, w_2, w_3$ are tuneable weights based on application requirements (e.g., $w_1$ is high for real-time chat, $w_2$ is high for offline batch processing).
- $\text{Latency}(p)$ is the moving average of Time to First Token (TTFT).
- $\text{Price}(p)$ is the normalized blended cost of input/output tokens.
- $\text{ErrorRate}(p)$ is the percentage of recent 429/500 errors.

The gateway routes the request to $\arg\min_{p} Cost(p)$.

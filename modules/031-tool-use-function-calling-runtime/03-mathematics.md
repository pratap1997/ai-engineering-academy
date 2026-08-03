# Mathematics of Tool Runtimes

While tool execution is heavily engineering-focused, several mathematical concepts underpin its design, particularly in rate limiting and risk assessment.

## 1. Token Bucket Rate Limiting

The Token Bucket algorithm is the industry standard for rate limiting API calls and tool executions.

Let $C$ be the maximum capacity of the bucket (maximum burst size).
Let $R$ be the refill rate (tokens added per second).
Let $T(t)$ be the number of tokens in the bucket at time $t$.

The number of tokens at time $t_2$ given the previous state at $t_1$ is:

$$T(t_2) = \min(C, T(t_1) + R \cdot (t_2 - t_1))$$

When a tool requires $k$ tokens to execute:
- If $T(t) \ge k$, the execution is allowed, and $T(t) \leftarrow T(t) - k$.
- If $T(t) < k$, the execution is denied (Rate Limited).

## 2. Tool Selection Entropy $H(T|q)$

When evaluating how well an LLM selects tools, we can look at the entropy of the tool distribution given a query $q$. Let $T$ be the set of available tools, and $p(t_i|q)$ be the probability the model selects tool $t_i$ for query $q$.

$$H(T|q) = - \sum_{i=1}^{|T|} p(t_i|q) \log_2 p(t_i|q)$$

- High Entropy: The model is confused or the query is highly ambiguous; it assigns similar probabilities to many tools.
- Low Entropy: The model is highly confident in selecting one or a few specific tools.

## 3. Risk Scoring Function $R(a)$

To implement dynamic security, we calculate a risk score $R(a)$ for an action $a$.
An action consists of a tool $t$ and its arguments $X = \{x_1, x_2, \dots, x_n\}$.

$$R(a) = W_t \cdot \text{BaseRisk}(t) + \sum_{i=1}^{n} W_{arg} \cdot \text{ArgRisk}(x_i, t)$$

Where:
- $\text{BaseRisk}(t)$ is the inherent danger of the tool (e.g., `read_file` = 0.2, `execute_bash` = 0.9).
- $\text{ArgRisk}(x_i, t)$ evaluates the danger of specific arguments (e.g., `execute_bash` with `rm -rf` has a high arg risk).
- $W_t, W_{arg}$ are learned or configured weights.

If $R(a) > \tau$ (where $\tau$ is a threshold), the runtime triggers human-in-the-loop approval or blocks the action entirely.

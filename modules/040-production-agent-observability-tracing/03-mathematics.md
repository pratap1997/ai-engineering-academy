# 03 - Mathematics: Tracing, Cost, and Latency

## Trace Tree Hierarchy

A trace $T$ can be represented as a directed acyclic graph (specifically, a tree if there's a single root and no shared children), defined by a set of Spans $S$ and directed edges $E_{parent}$ representing parent-child relationships.

For a span $s \in S$, its duration is $\Delta t^{(s)} = t_{end}^{(s)} - t_{start}^{(s)}$.

## Token and Cost Accounting Function

The total cost of a trace is the sum of costs for all spans that involve an LLM call. Let $P_{prompt}$ be the price per prompt token and $P_{completion}$ be the price per completion token for the specific model used in span $s$.

$$ C(T) = \sum_{s \in S_{LLM}} \left( N_{prompt}^{(s)} \cdot P_{prompt}^{(s)} + N_{completion}^{(s)} \cdot P_{completion}^{(s)} \right) $$

Where $S_{LLM} \subset S$ is the subset of spans representing LLM completions.

## Latency Quantile Estimations

In production, average latency is a poor metric due to heavy-tailed distributions. We care about the 95th (P95) and 99th (P99) percentiles. For streaming data without storing all elements, we approximate these using a basic form of **Reservoir Sampling** or digest algorithms.

Given a sorted array of $N$ latencies $L = [l_1, l_2, \dots, l_N]$, the $p$-th percentile value (where $0 < p < 1$) is at index $k = p \cdot N$.
If $k$ is not an integer, we interpolate between $L_{\lfloor k \rfloor}$ and $L_{\lceil k \rceil}$.

For real-time profiling, maintaining a bounded sorted list of recent latencies allows for sliding-window P95 calculations:
$$ P95 \approx \text{Percentile}(L_{window}, 0.95) $$

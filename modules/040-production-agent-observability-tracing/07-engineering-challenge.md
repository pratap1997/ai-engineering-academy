# 07 - Engineering Challenge

## Real-Time Agent Cost & Latency Anomaly Alerting System

**Task**: Build a monitoring system that wraps around an agent framework to detect anomalies in real-time.

**Requirements**:
1. Implement a wrapping `Tracer` that intercepts all LLM calls.
2. Track running cost per session/user.
3. If a single trace exceeds $0.50 in LLM costs, immediately emit a `CostAnomalyEvent`.
4. Implement a sliding window of the last 50 LLM call latencies. If the current latency exceeds the P95 latency by 2x, emit a `LatencyAnomalyEvent`.

**Constraints**:
- Use only the standard Python library.
- Must support hierarchical spans (e.g., Tool Call -> LLM Call).

No hints provided. Validate your implementation using simulated traffic patterns.

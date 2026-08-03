# Real-World Applications

Benchmarking and evaluating agents is critical in production systems.

## SWE-agent Benchmark Testing
Researchers at Princeton developed SWE-agent, an agentic loop designed to solve GitHub issues. They evaluate SWE-agent continuously against the SWE-bench dataset. For every change to the agent's prompt or tools, they run the agent against a subset of SWE-bench (e.g., SWE-bench Lite) to ensure no regressions occur. This is CI/CD for agentic logic.

## LangSmith Evaluation Pipelines
Platforms like LangSmith allow teams to log entire agent trajectories. They use LLM-as-a-judge to grade these trajectories asynchronously. If an agent goes off-topic in a customer support conversation, an evaluator model flags it, computes the score, and alerts the engineering team.

## OpenAI Evals Framework
OpenAI open-sourced `evals`, a framework they use internally to benchmark new models (like GPT-4) before release. It contains templates for evaluating tool-use, reasoning, and coding capabilities across hundreds of carefully curated datasets.

## Anthropic Internal Agent Benchmarks
Anthropic tests Claude's capabilities using rigorous internal benchmarks that test long-context recall (Needle in a Haystack) and multi-step tool use. By wrapping these in automated testing suites, they can calculate Pass@K across various temperature settings to find the optimal deployment configuration.

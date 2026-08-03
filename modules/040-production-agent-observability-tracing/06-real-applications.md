# 06 - Real Applications

## 1. LangSmith Tracing & Evaluation
LangSmith is a platform built specifically for LLM applications. It provides deep visibility into chains and agents.
- **Tracing**: Automatically captures hierarchical traces of LangChain/LangGraph executions.
- **Evaluation**: Allows running datasets against agent traces to calculate metrics like helpfulness or hallucination rates.

## 2. Arize Phoenix & Weights & Biases Prompts
These platforms focus on LLM observability, providing tools to track prompt iterations, embeddings, and generative responses.
- **Arize Phoenix**: Offers local observability, visualizing traces and enabling troubleshooting of RAG systems.
- **W&B Prompts**: Integrates with ML lifecycle tools to version control prompts and track their performance over time.

## 3. OpenTelemetry GenAI Semantic Conventions
Standardization is key. OpenTelemetry is defining semantic conventions specifically for Generative AI.
- Specifies standard attributes like `gen_ai.system`, `gen_ai.request.model`, `gen_ai.response.choices`.
- Ensures that traces emitted by LangChain, LlamaIndex, or custom agents can be consumed by standard backends like Datadog, Honeycomb, or New Relic.

## 4. Datadog LLM Observability
Datadog has expanded its APM to include LLMs, correlating AI traces with underlying infrastructure metrics.
- Useful for detecting if high latency is caused by the LLM provider or by an internal database in a RAG setup.

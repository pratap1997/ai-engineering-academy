# 08 - Assessment

1. **What is a Span in distributed tracing?**
   - A single, timed operation within a trace.

2. **Why is traditional APM insufficient for LLM agents?**
   - It lacks visibility into prompts, completions, and token costs which are highly variable.

3. **What is the purpose of OpenTelemetry GenAI Semantic Conventions?**
   - To provide standardized attribute names (like `gen_ai.request.model`) across different observability tools.

4. **In the cost formula $C(T)$, why are prompt and completion tokens priced differently?**
   - Because generation (decoding) is more computationally expensive than prompt processing (prefilling).

5. **What does P95 latency mean?**
   - The latency value that 95% of requests fall below.

6. **How does a Tracer maintain parent-child relationships?**
   - By keeping track of the active context/span and assigning the current span's ID as the parent ID for new spans.

7. **What is the "Black Box" metaphor in agent tracing?**
   - Recording every input, output, and decision to reconstruct agent behavior after a failure.

8. **Why track token usage at the Span level?**
   - To identify exactly which sub-task or tool loop is consuming the budget.

9. **What is Reservoir Sampling used for in this context?**
   - Estimating percentiles (like P95) efficiently in a streaming environment without unbounded memory growth.

10. **How can you trace a multi-agent system?**
    - By propagating the `Trace ID` across agent boundaries and linking their respective root spans.

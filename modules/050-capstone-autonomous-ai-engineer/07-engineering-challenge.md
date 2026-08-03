# Engineering Challenge: End-to-End System Integration

Your challenge is to expand the `AutonomousAIEngineerSystem` to support **dynamic sub-agent spawning** and **self-healing DAG workflows**.

### The Requirements
1. **Dynamic Swarm**: Instead of a static workflow, the `TaskPlanner` must dynamically spawn agents via the `MultiAgentMessageBus` based on the task description (e.g., spawning a `DatabaseAdmin` agent if SQL is detected).
2. **Self-Healing DAG**: Modify the `DAGWorkflowEngine` so that if an action fails (e.g., tests fail), it dynamically injects a "fix_code" node into the DAG and re-attempts the execution loop up to $N$ times.
3. **Trace Export**: Write a JSON exporter for the `OpenTelemetryTracer` that outputs a Chrome Trace Format (CTF) compatible file, so execution traces can be visualized in `chrome://tracing`.

### Constraints
- No external libraries.
- Standard library only.
- Must not deadlock during infinite self-healing loops.

Good luck, Systems Architect.

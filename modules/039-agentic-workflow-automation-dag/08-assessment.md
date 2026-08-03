# 08 - Assessment

1. **What is the primary mathematical structure used to model agentic workflows?**
   - Directed Acyclic Graph (DAG)

2. **Why must the graph be acyclic?**
   - To guarantee termination and prevent infinite loops in automated execution.

3. **Which algorithm is commonly used to find an execution order for a DAG?**
   - Kahn's Algorithm (or Depth-First Search for topological sorting).

4. **What is a Human-in-the-Loop (HITL) gate?**
   - A mechanism that pauses workflow execution at a specific node, waiting for explicit external approval before resuming.

5. **In the context of workflows, what is checkpointing?**
   - Saving the global state of the workflow at a specific node so that it can be restored later.

6. **Why is checkpointing useful?**
   - It allows recovery from failures without having to restart the entire workflow from the beginning.

7. **What is exponential backoff?**
   - A retry strategy where the wait time between retries increases exponentially to prevent overloading a failing resource.

8. **What does the transition function $\delta: S \times A \to S$ represent?**
   - The process of a workflow node taking the current state and an action to produce a new modified state.

9. **If node A depends on node B, and node B depends on node C, what is the topological order?**
   - C, B, A

10. **How does Temporal.io provide durable execution?**
    - By using event sourcing to log all state transitions, allowing it to replay history and resume execution after a crash.

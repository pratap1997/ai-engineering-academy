# Assessment: Multi-Agent Orchestration

## Questions

### Conceptual
1. Why is a multi-agent system often preferable to a single, monolithic LLM with a massive context window?
2. Explain the difference between a Star topology and a Mesh topology in agent communication.
3. In what scenario would an adversarial/debate pattern be strictly superior to a hierarchical supervisor-worker pattern?

### Mathematical
4. Given 8 agents in a fully connected Mesh topology, what is the exact number of messages generated if every agent broadcasts exactly one message to all peers?
5. Define Nash Equilibrium in the context of two competitive agents debating a topic.
6. According to Condorcet's paradox, is a Condorcet winner guaranteed to exist in agent voting? Why or why not?

### Implementation
7. When implementing an event-driven `MessageBus`, why is it critical that agents clear their inbox upon calling `receive()`?
8. How does implementing "backpressure" in a Supervisor agent prevent system failure?

### Judgment
9. You are building an agentic system to write a complete fantasy novel. Which orchestration pattern (Hierarchical, Collaborative, Competitive) do you choose and why?
10. A peer-to-peer agent system with 20 agents is consistently timing out due to context window exhaustion. What is the most likely architectural flaw, and how do you fix it?

---

## Debrief & Answers

1. **Answer**: It distributes cognitive load, allows for specialization (custom system prompts and isolated toolsets per agent), and prevents early hallucination compounding through independent verification.
2. **Answer**: In a Star topology, all messages route through a central supervisor node (O(n) complexity). In a Mesh topology, all agents communicate directly with one another (O(n^2) complexity).
3. **Answer**: When the ground truth is ambiguous or requires fact-checking. Debate forces models to defend their logic, which inherently minimizes hallucinations, whereas hierarchical workers might just confidently hallucinate and the supervisor might blindly accept it.
4. **Answer**: 56 messages. Each of the 8 agents sends a message to the other 7 agents. $8 \times 7 = 56$.
5. **Answer**: A state in the debate where neither agent can change its argumentation strategy to achieve a higher reward (e.g., winning the debate), assuming the other agent's strategy remains fixed.
6. **Answer**: No. Preferences can be cyclical (A beats B, B beats C, C beats A), meaning no single option beats all others in head-to-head matchups.
7. **Answer**: To prevent reprocessing the same messages in subsequent steps, which would lead to infinite loops or duplicate task execution.
8. **Answer**: Backpressure prevents a fast sender (Supervisor) from overwhelming a slow receiver (Worker). Without it, the worker's inbox grows infinitely, eventually causing out-of-memory errors or massive latency.
9. **Answer**: Hierarchical. A novel requires structured breakdown (Manager -> Chapter Outliners -> Scene Writers -> Editors). A pure mesh would result in chaotic, disjointed story elements.
10. **Answer**: The system is likely using a pure Mesh topology with excessive broadcasting, leading to $O(n^2)$ message complexity which rapidly fills context windows. Fix: Implement a Star or Hierarchical topology, or restrict agents to only subscribe to specific topics rather than receiving all broadcasts.

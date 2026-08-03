# Mental Model: Multi-Agent Systems

## The "Team of Experts" Metaphor
To effectively reason about multi-agent systems, adopt the "team of experts" metaphor. Instead of viewing the system as a single omniscient brain, view it as a corporate department. There is a **supervisor** who understands the high-level goal, and **specialists** (workers) who possess deep knowledge in narrow domains. 

Just as a human engineering team operates, these agents must communicate, hand off tasks, and resolve disagreements. The architecture of your multi-agent system dictates how this corporate department functions.

## Communication: Passing Messages in a Shared Inbox
In a multi-agent system, communication is fundamentally about message passing. Agents do not share a single "mind"; they share an inbox or an event bus. 

When Agent A needs Agent B to perform a task, it sends a discrete message containing the context and the request. Agent B processes the message and replies. This decoupling ensures that agents can operate asynchronously, mirroring real-world distributed systems.

## State: The Blackboard Pattern
While message passing is excellent for direct requests, complex multi-agent systems often employ a **Blackboard** pattern for shared state. 
Imagine a literal blackboard in a meeting room. All agents can read from and write to this blackboard. As the state evolves (e.g., a software architecture document is iteratively drafted), agents observe the changes and contribute their specialized input when appropriate.

## Orchestration Topologies

The flow of messages and authority in a multi-agent system defines its topology.

### 1. Star (Centralized)
```
      [Worker A]
          ^
          |
[Worker B] <--> [Supervisor] <--> [Worker C]
```
All communication routes through a central supervisor. The supervisor maintains total control, assigning tasks and synthesizing results.
- **When to use**: Highly structured tasks where subtasks are independent (e.g., MapReduce style processing).

### 2. Ring
```
[Agent A] ---> [Agent B]
    ^              |
    |              v
[Agent D] <--- [Agent C]
```
Agents process information sequentially, passing the result to the next agent in the pipeline.
- **When to use**: Sequential pipelines where each step strictly depends on the previous (e.g., Write -> Review -> Deploy).

### 3. Mesh (Peer-to-Peer)
```
[Agent A] <--> [Agent B]
    ^  \        /  ^
    |   \      /   |
    v    \    /    v
[Agent C] <--> [Agent D]
```
All agents can communicate with all other agents directly.
- **When to use**: Complex, open-ended problem solving where dynamic collaboration is needed (e.g., a simulated debate or brainstorming session).

### 4. Hierarchical Tree
```
          [Manager]
         /         \
    [Lead A]     [Lead B]
     /    \       /    \
  [W1]   [W2]  [W3]   [W4]
```
A deep structure where supervisors manage sub-supervisors.
- **When to use**: Massive, multi-faceted tasks that require recursive decomposition (e.g., writing an entire software application from scratch).

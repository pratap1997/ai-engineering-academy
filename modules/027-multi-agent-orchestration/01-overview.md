# Module 027 - Multi-Agent Orchestration

## Overview
Multi-agent systems represent a paradigm shift from monolithic single-agent architectures to distributed, specialized networks. As AI applications tackle increasingly complex tasks, one agent is often not enough. Multi-agent orchestration distributes cognitive load, enables specialization, and allows systems to self-correct through debate and review.

### Why One Agent is Not Enough
A single LLM acting as a monolithic agent faces severe limitations:
1. **Context limits**: A single agent cannot effectively maintain deep context across vastly different domains simultaneously.
2. **Hallucination compounding**: Errors in early reasoning steps propagate and magnify without independent verification.
3. **Sequential bottlenecking**: Single agents process tasks linearly, whereas many real-world tasks (like code review, security auditing, and testing) are naturally parallelizable.

### Orchestration Patterns

We can categorize multi-agent systems into three primary orchestration patterns:

1. **Hierarchical (Supervisor → Worker)**
   - A central supervisor agent breaks down a task, delegates subtasks to specialized worker agents, and synthesizes the results.
   - *Example*: An engineering manager agent delegating frontend and backend tasks to specialist agents.

2. **Collaborative (Peer-to-Peer)**
   - Agents operate as equals, passing messages in a shared context or blackboard. They collaborate dynamically based on their specific triggers and roles.
   - *Example*: A group chat of agents where each contributes when its domain expertise is relevant.

3. **Competitive (Adversarial/Debate)**
   - Agents take opposing viewpoints or roles (e.g., generator vs. critic) to refine outputs through debate and critique, minimizing hallucinations.
   - *Example*: A red-teaming scenario where an attacker agent tries to bypass a defender agent.

### Framework Implementations
Several modern frameworks implement these patterns:
- **AutoGen**: Excels at conversational, collaborative peer-to-peer patterns and debate structures.
- **LangGraph**: Provides state-machine-based, rigid hierarchical and cyclical orchestration with fine-grained control over the flow.
- **OpenAI Swarm**: A lightweight framework focusing on seamless handoffs between specialized agents, primarily following a peer-to-peer or loose hierarchical model.

### Prerequisites
Before diving into multi-agent systems, a firm grasp of **Module 026 (Agentic Loop Primitives)** is strictly required. Understanding single-agent ReAct loops, tool usage, and basic memory management is foundational to orchestrating multiple agents safely.

### Real Motivation
The true motivation for multi-agent systems lies in tasks that are naturally parallel and require distinct perspectives. For instance, in software development, having the same agent write code and review its own code is an anti-pattern. Distinct agents with varied system prompts (e.g., Coder vs. Security Reviewer) provide the necessary friction to produce high-quality, robust outcomes.

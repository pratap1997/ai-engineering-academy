# 01 - Overview: Agentic Workflow Automation & DAGs

## Motivation
As AI systems become more complex, they move from linear, single-shot deterministic tasks to dynamic, multi-step agentic workflows. To ensure reliability, these workflows must be modeled as Directed Acyclic Graphs (DAGs). DAGs provide a robust engine for executing dependent tasks, handling state transitions, retrying failures, and incorporating human-in-the-loop (HITL) approval gates.

## Core Concepts
- **Deterministic vs Agentic Workflows**: Traditional workflows execute predefined steps. Agentic workflows dynamically route based on state but still require a structured graph (DAG) to prevent infinite loops and ensure termination.
- **DAG Execution Engine**: An execution engine that processes nodes in a topologically sorted order, ensuring dependencies are satisfied before execution.
- **State Graph Transitions**: The system maintains a global state. Each node represents a transition function that modifies the state and passes it to the next node.
- **Human-in-the-Loop (HITL) Approval Gates**: Workflows can pause at critical nodes, requiring explicit human approval before resuming, preventing autonomous systems from taking irreversible actions.
- **Checkpointing & Time-Travel Recovery**: Saving state at each node allows the system to rollback to previous states upon failure and resume from the exact point of interruption.

## Prerequisites
- 026-agentic-loop-primitives
- 027-multi-agent-orchestration

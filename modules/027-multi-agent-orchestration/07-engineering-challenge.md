# Engineering Challenge: Build a Peer-Review Multi-Agent System

## The Challenge
Your task is to build a highly robust, asynchronous Peer-Review multi-agent system from scratch using Python (no external ML frameworks).

## Requirements

### 1. Agents
You must implement three distinct, specialized agents:
* **Coder**: Receives a spec and outputs code.
* **Reviewer**: Evaluates code for style and correctness.
* **Security Auditor**: Evaluates code exclusively for vulnerabilities.

### 2. Mechanics
* **Asynchronous Message Passing**: Agents must not block one another. They should read from an inbox, process, and write to an outbox/bus.
* **Quorum-based Acceptance**: Code is only accepted if a quorum is reached. Specifically, 2 out of the 3 agents (typically the two reviewers) must explicitly agree the code is ready to ship.
* **Backpressure**: The Supervisor must implement backpressure. If a worker's inbox queue exceeds 5 pending messages, the supervisor must pause assigning new tasks to that worker.

### 3. Success Criteria
Your implementation must pass a deterministic integration test:
* Feed a pipeline of 10 distinct tasks into the system.
* The system must process all 10 tasks.
* The system must achieve a <5% error rate (simulate agent responses using a deterministic mock function provided in the starter code).

**Constraint:** Do not use `asyncio`; implement the asynchronous behavior using a discrete event loop (stepping agents turn-by-turn).

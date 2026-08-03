# Overview: Agent Evaluation & Benchmarking

Evaluating autonomous agents is significantly more complex than evaluating standard machine learning models. A traditional model (like an image classifier) takes an input and produces an output in a single, deterministic step. An agent, however, takes an input, plans a sequence of actions, interacts with an environment, receives feedback, and updates its state over time.

## Why Evaluating Agents is Hard

1. **Nondeterminism**: Due to sampling parameters (temperature) and dynamic environments (APIs changing, network latency), agents can produce different trajectories for the exact same initial state.
2. **Multi-Step Trajectories**: A single failure at step 2 of a 10-step plan might doom the entire operation, or the agent might recover. We need to evaluate not just the final outcome, but the efficiency and safety of the trajectory.
3. **Environment State Changes**: Agents take actions that modify the world (writing files, dropping database tables). Evaluation environments must be sandboxed and perfectly reproducible.

## Static vs Dynamic Evaluation

- **Static Evaluation**: Checking if the agent outputs the correct string or JSON payload at a single step.
- **Dynamic Evaluation**: Instantiating an environment (e.g., a Docker container), letting the agent run its loop, and then asserting on the final state of the environment.

## Key Benchmarks

- **SWE-bench**: Evaluates an agent's ability to resolve real-world GitHub issues in Python repositories. It tests the agent's ability to search codebases, understand context, write patches, and run tests.
- **GAIA**: A benchmark requiring general AI assistants to reason, use tools, and browse the web to answer complex questions.
- **AgentBench**: A framework to evaluate LLMs as agents across diverse environments like OS interaction, database management, and web browsing.

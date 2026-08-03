# Module 046: Agentic Planning & Tree Search (MCTS)

## Overview
This module explores advanced planning algorithms for LLM-based agents, focusing on Tree of Thoughts (ToT) and Monte Carlo Tree Search (MCTS). 

Linear Chain-of-Thought reasoning limits an agent to a single exploration path. In contrast, Tree-of-Thoughts enables deliberate reasoning by branching into multiple possibilities. MCTS goes further by incorporating a formal search heuristic (like UCT) and four systematic phases:
1. **Selection:** Traverse the tree to a promising leaf node.
2. **Expansion:** Generate new possible next steps.
3. **Simulation/Rollout:** Simulate the outcome of the steps to evaluate their potential.
4. **Backpropagation:** Update the tree nodes with the success metrics of the simulated path.

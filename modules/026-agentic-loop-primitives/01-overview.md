# 01 - Overview: Agentic Loop Primitives

## Context and Motivation
Large Language Models (LLMs) like GPT-4 or Claude have historically functioned as passive text generators: given a prompt, they predict the most likely continuation. While powerful, this paradigm is inherently limited. Passive models cannot independently browse the web for up-to-date information, execute code to solve complex math equations, or interact with external APIs to complete tasks on behalf of a user. 

To transition from passive text generators to active **AI Agents**, we introduce **Agentic Loop Primitives**. These primitives transform an LLM from a simple input-output function into a persistent reasoning engine capable of operating autonomously over extended periods. This module focuses on the fundamental building blocks of autonomous agents: the Perceive-Think-Act cycle, the ReAct framework, Tool Use (function calling), and Planning strategies.

By bridging the gap between reasoning and acting, agentic loops allow models to interact with the world, observe the consequences of their actions, and iteratively adjust their plans to achieve complex, long-term goals.

## The Perceive-Think-Act Cycle
At the core of any agentic system is the loop. This cycle, inspired by cognitive architectures and reinforcement learning, typically consists of:
1. **Perceive (Observe):** The agent receives input from its environment (e.g., user query, tool execution result, web page content).
2. **Think (Reason):** The agent processes the observation, updates its internal state, and decides what to do next. This step relies heavily on the LLM's emergent reasoning capabilities, often enhanced by techniques like Chain-of-Thought.
3. **Act:** The agent executes an action (e.g., generating a final response, calling an external tool, or emitting a structured command).

## The ReAct Framework (Reasoning + Acting)
Introduced by Yao et al. in 2022, **ReAct** explicitly synergizes reasoning and acting within language models. Before ReAct, agents typically either reasoned (e.g., Chain-of-Thought) without the ability to interact, or acted (e.g., reinforcement learning agents) without explicitly articulated reasoning. ReAct instructs the LLM to generate alternating "Thought" and "Action" trajectories. By explicitly writing out its thought process *before* taking an action, the agent can maintain focus, track its progress, and recover from errors when an action yields an unexpected observation.

## Tool Use (Function Calling)
Tool use is the mechanism by which agents execute actions. Instead of relying solely on its internal weights, an agent can be provided with a registry of tools (e.g., a calculator, a Python interpreter, an SQL executor). The agent learns to recognize when its internal knowledge is insufficient and explicitly outputs a structured request to invoke a specific tool. Tool use fundamentally expands the capability surface of AI systems, allowing them to perform exact arithmetic, fetch real-time data, and interact with software systems.

## Planning Strategies
For tasks that cannot be solved in a single step, agents require planning. Simple approaches include **Task Decomposition**, where a large goal is broken into sequential sub-tasks. More advanced planning strategies view the problem as a search space. Techniques like **Tree of Thoughts (ToT)** allow agents to explore multiple reasoning paths simultaneously, evaluate intermediate states, and backtrack if a path proves unpromising.

## Prerequisites
Before diving into agentic loops, you should have a strong understanding of:
- **Module 010 (Attention):** Understanding how models contextualize information.
- **Module 011 (Transformers):** The underlying architecture powering modern LLMs.
- **Module 012 (BERT/GPT Pretraining):** How these models acquire their foundational knowledge.
- **Modules 013 & 014 (Tokenization & Positional Encodings):** The mechanics of how text is processed.

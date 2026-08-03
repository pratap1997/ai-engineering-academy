# 02 - Mental Model: Agentic Loop Primitives

## The Agent Loop as a Finite State Machine
To intuitively grasp how an AI agent operates, it is helpful to conceptualize it as a Finite State Machine (FSM) transitioning through a continuous cycle. Unlike a standard LLM call which simply maps an input to an output, an agentic loop operates as a persistent process:

1. **Observe (State 1):** The agent gathers context. This could be the initial user prompt, or the result of a previous action.
2. **Think (State 2):** The agent analyzes the observation. It determines if it has enough information to solve the problem or if it needs to take an action.
3. **Act (State 3):** The agent decides to either emit a final answer to the user (terminating the loop) or call a tool (transitioning to State 4).
4. **Tool Execution (State 4):** The external environment executes the tool and returns the result, which becomes the next Observation.

```text
+-------------------+      +-------------------+
|                   |      |                   |
| 1. OBSERVE        +----->+ 2. THINK          |
| (Environment/Input|      | (Chain of Thought)|
|                   |      |                   |
+---------^---------+      +---------+---------+
          |                          |
          |                          |
+---------+---------+      +---------v---------+
|                   |      |                   |
| 4. TOOL EXECUTION |<-----+ 3. ACT            |
| (Python, Web, API)|      | (Select Tool/Arg) |
|                   |      |                   |
+-------------------+      +-------------------+
```
*Figure 1: The Perceive-Think-Act Cycle*

## Comparison to Human Problem-Solving
Consider how a human solves a complex math problem or writes a research paper. We do not generate the final answer in one continuous stream of consciousness. Instead, we use a loop:
- **Read:** We read the question (Observe).
- **Think:** We realize we don't know the exact value of $15.4 \times 89.2$ off the top of our head (Think).
- **Act:** We pull out a calculator and punch in the numbers (Act).
- **Check Result:** We read the result from the screen (Observe), and then continue our derivation (Think).

Agentic loops endow LLMs with this exact capability, breaking them free from the constraint of having to "know" everything entirely within their pre-trained weights.

## The Scratchpad Metaphor
A crucial component of the mental model is the **Scratchpad**. Because LLMs are stateless by default, the "memory" of the loop is maintained by concatenating the entire history of Observations, Thoughts, and Actions into a single, growing prompt (the scratchpad). 

When the agent reaches the "Think" step of iteration 5, it sees the entire trajectory of iterations 1 through 4. The scratchpad acts as the agent's working memory, providing context and preventing it from repeating failed actions. 

## Tool Use as Function Dispatch
Think of Tool Use not as the LLM executing code, but as **Function Dispatch**. The LLM acts as the orchestrator. It writes the exact signature and arguments of the function to be called, but execution happens in a secure sandbox or runtime external to the model. 

```text
Agent (LLM)                        Runtime (Environment)
+----------------+                 +-------------------+
| Decides to     |                 |                   |
| calculate math |--- call_math -->| Executes 5 * 10   |
|                |                 |                   |
|                |<--- 50 ---------| Returns result    |
+----------------+                 +-------------------+
```
*Figure 2: Tool Call Flow*

## Planning as Tree Search
For simple tasks, linear Chain-of-Thought is sufficient. For complex tasks, linear reasoning often hits dead ends. The mental model for advanced planning is **Tree Search**.

Imagine a maze. A standard LLM tries to draw a single line from start to finish without lifting its pen. If it hits a wall, it fails. 
An agent with tree search planning acts like a pathfinding algorithm. It explores a branch, evaluates its current position (State Evaluation), realizes it's near a wall, and deliberately backtracks to a previous node to try a different path. This transforms the agent from a reactive sequence-generator into a deliberate problem-solver.

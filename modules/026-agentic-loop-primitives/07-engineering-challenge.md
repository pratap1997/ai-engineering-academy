# Engineering Challenge: Build a Multi-Tool ReAct Agent for Math + Code

## Objective
Implement an advanced ReAct Agent capable of solving complex multi-step problems requiring dynamic tool selection, context management, and error recovery. You will build upon the basic ReAct loop by introducing advanced heuristics and robustness features.

## Requirements

1. **Tool Priority Ordering**
   - You must implement at least two tools: a `Calculator` (for simple math) and a `PythonExecutor` (for complex logic or loops).
   - Your agent's internal reasoning or registry must enforce a priority heuristic: it must prefer the `Calculator` over `PythonExecutor` for simple arithmetic to save computational overhead.

2. **Trajectory Compression**
   - The LLM context window is finite. Implement a `compress_trajectory(trajectory)` method that summarizes older steps when the trajectory exceeds 5 steps. 
   - The prompt must seamlessly integrate this compressed summary along with recent exact steps.

3. **Retry Logic (Error Recovery)**
   - If a tool returns an error (e.g., SyntaxError in Python, or invalid math expression), the agent must not immediately fail.
   - Implement logic to retry failed tool calls up to **3 times** with modified arguments based on the error observation.

4. **Task Execution**
   - Your agent must solve a task requiring a minimum of **5 sequential tool calls**.
   - Example Task: *"Calculate the sum of the first 10 prime numbers, then multiply that sum by the square root of 1024, and finally add 50."*

## Success Criteria
- **Measurable Goal**: The agent successfully solves the 5-step task.
- **Efficiency Constraint**: The task must be completed in **≤8 steps**.
- **Accuracy Constraint**: The final output must be 100% mathematically accurate.
- **Robustness**: Your test suite must include a scenario where a tool intentionally fails on the first attempt, and the agent successfully recovers via the retry logic.

*Note: You are expected to implement this challenge from scratch without the use of external frameworks like LangChain or AutoGen.*

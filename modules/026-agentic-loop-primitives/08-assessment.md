# Assessment: Agentic Loop Primitives

## Questions

### Conceptual
1. **What is the primary architectural difference between Chain-of-Thought (CoT) prompting and ReAct?**
2. **In a ReAct loop, what is the role of an "Observation" and how does it affect the agent's trajectory?**
3. **Why is parsing the LLM output into strict structural components (e.g., Thought, Action, Action Input) critical for autonomous agents?**

### Mathematical
4. **Given a UCB1 algorithm with an exploration constant $c = \sqrt{2}$, derive the score for a tool that has been pulled 4 times with 3 successes, given a total of 100 tool pulls across the system.**
5. **Assume an agent requires exactly $k$ steps to solve a problem. If the probability of a tool failing at any step is $p$, and a failure terminates the run, what is the probability that the agent successfully completes the task?**
6. **Suppose a task decomposer breaks a problem into $n$ subtasks. If the context window limit is $L$ tokens, and each trajectory step consumes $t$ tokens, formulate an inequality to determine the maximum number of subtasks an agent can solve before requiring trajectory compression.**

### Implementation
7. **Write the core logic for a `_parse_llm_output(output: str)` method in Python that uses regular expressions to extract the `Thought`, `Action`, and a JSON dictionary `Action Input` from a raw LLM string.**
8. **Implement a simple retry mechanism wrapper around a tool execution function that catches `Exception` and retries up to 3 times before returning a failure observation.**

### Judgment
9. **When would you choose to use a full ReAct loop over a simpler Chain-of-Thought approach? Provide a specific scenario.**
10. **Describe a scenario where a ReAct loop is highly likely to fail or enter an infinite loop. How would you architect the system to mitigate this?**

---

## Debrief & Model Answers

1. **CoT vs ReAct:** CoT generates a continuous stream of reasoning to reach an answer based entirely on the LLM's internal weights. ReAct interleaves reasoning (Thought) with actions that interact with an external environment, halting generation to wait for an external Observation before continuing.
2. **Observation Role:** The Observation injects grounded truth from the external environment (like search results or code output) back into the prompt. This grounds the agent's next Thought, allowing it to correct false assumptions or proceed with verified data.
3. **Strict Parsing:** It allows the orchestrator (the Python loop) to deterministically know when to stop the LLM generation and execute a function. Without strict formatting, the agent might hallucinate the tool's output instead of actually running the tool.
4. **UCB1 Derivation:** 
   Score = Exploitation + Exploration
   Exploitation = 3/4 = 0.75
   Exploration = $\sqrt{2} \times \sqrt{\ln(100) / 4} = \sqrt{2} \times \sqrt{4.605 / 4} = 1.414 \times 1.073 = 1.517$
   Total Score = 0.75 + 1.517 = 2.267
5. **Success Probability:** The agent must succeed $k$ times in a row. The probability is $(1 - p)^k$.
6. **Context Limit Inequality:** $n \times t < L$. Therefore, maximum $n = \lfloor L / t \rfloor$.
7. **Parser Implementation:** 
   ```python
   def _parse_llm_output(output):
       import re, json
       thought = re.search(r"Thought:\s*(.*?)\nAction:", output, re.DOTALL).group(1).strip()
       action = re.search(r"Action:\s*(.*?)\n", output).group(1).strip()
       action_input = json.loads(re.search(r"Action Input:\s*(.*?)$", output, re.DOTALL).group(1).strip())
       return thought, action, action_input
   ```
8. **Retry Wrapper:**
   ```python
   def execute_with_retry(tool_fn, max_retries=3, **kwargs):
       for attempt in range(max_retries):
           try:
               return tool_fn(**kwargs)
           except Exception as e:
               if attempt == max_retries - 1:
                   return f"Failed after {max_retries} attempts: {e}"
   ```
9. **When to use ReAct:** Use ReAct when the task requires up-to-date information (e.g., current stock prices), exact mathematics (e.g., multiplying large numbers), or interaction with external state (e.g., modifying a database). Use CoT for closed-domain reasoning or summarization.
10. **Failure Scenarios:** ReAct fails when a tool repeatedly returns an unhelpful error (e.g., a generic `SyntaxError` with no context), causing the agent to repeatedly try the same failing action. Mitigation includes implementing a retry limit, forcing the agent to try a different tool, or using a "system prompt" intervention warning the agent that it is repeating itself.

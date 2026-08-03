# Assessment

1.  **Concept:** What is the primary purpose of a Tool Runtime in an agentic system? (Explain using the OS Kernel metaphor).
2.  **Concept:** Describe the difference between "Prompt-based security" and "Execution Sandboxing". Which is more reliable and why?
3.  **Concept:** Why is it dangerous to allow an LLM to directly execute `eval()` or `exec()` on Python code it generates?
4.  **Math:** A Token Bucket rate limiter has a capacity of $C=10$ and a refill rate of $R=2$ tokens/second. The bucket currently has $5$ tokens. A burst of 8 tool calls arrives simultaneously (each costing 1 token). How many are executed, and how many are rate-limited?
5.  **Math:** In the same token bucket, after the burst in Q4, exactly 3 seconds pass with no activity. How many tokens are in the bucket?
6.  **Math:** A risk scoring function uses $R(a) = 0.5 \cdot \text{Base}(t) + 0.5 \cdot \text{Arg}(x, t)$. If $\text{Base}(\text{bash}) = 1.0$ and $\text{Arg}(\text{rm -rf}, \text{bash}) = 1.0$, what is the risk score? If the threshold $\tau = 0.8$, is the action blocked?
7.  **Implementation:** How can Python's `ast` module be used to create a SafePythonExecutor?
8.  **Implementation:** Explain how to implement an Audit Log that prevents an agent from erasing its own history.
9.  **Judgment:** You are building an agent that needs to format dates using a Python script. Should you give it access to a full bash terminal, or a restricted Python AST evaluator? Why?
10. **Judgment:** If an agent receives a "Rate Limit Exceeded" error from the runtime, how should the runtime format the response to help the agent recover?

## Answers (Abridged)
1. The Tool Runtime acts as the trusted kernel, mediating between the untrusted LLM (user space) and the host environment, enforcing security, limits, and providing safe execution.
2. Prompt security relies on the LLM obeying instructions (highly fallible). Sandboxing enforces hardware/OS-level restrictions regardless of the LLM's output. Sandboxing is far more reliable.
3. `eval()` executes arbitrary code. The LLM could generate malicious code (e.g., `os.system('rm -rf /')`) that compromises the host system.
4. The bucket has 5 tokens. 5 calls execute, 3 are rate-limited.
5. Remaining tokens: 0. After 3 seconds, $0 + (2 \times 3) = 6$ tokens.
6. $R(a) = 0.5(1.0) + 0.5(1.0) = 1.0$. Since $1.0 > 0.8$, the action is blocked.
7. `ast.parse()` converts code to an Abstract Syntax Tree. We can walk the tree (`ast.NodeVisitor`) and reject the execution if we encounter forbidden nodes (like `Import`, `Call` to `eval`, or file I/O operations).
8. By ensuring the Audit Log object only exposes an `append()` or `log()` method to the tool execution layer, and storing the underlying data structure outside the execution sandbox environment.
9. Restricted Python AST evaluator. Principle of Least Privilege. Date formatting does not require OS-level access.
10. The response should clearly state the error and provide a retry-after time, e.g., `{"error": "RateLimitExceeded", "retry_in_seconds": 5}`.

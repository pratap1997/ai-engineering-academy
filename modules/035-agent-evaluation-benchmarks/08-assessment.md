# Assessment: Agent Evaluation

**1. Why is deterministic testing (like standard unit testing) often insufficient for evaluating agents?**
Agents often take multi-step trajectories and interact with changing environments. A deterministic test only checks the final output, ignoring the efficiency, safety, and reasoning path of the trajectory.

**2. What does Pass@K measure?**
It measures the probability that an agent will successfully complete a task if it is given K independent attempts.

**3. In the Pass@K formula, what happens if $K$ is greater than $N - c$?**
If the number of failures ($N - c$) is less than $K$, the formula evaluates to a 100% pass probability, as at least one of the $K$ attempts is guaranteed to be a success.

**4. What is SWE-bench?**
A benchmark dataset of real-world GitHub issues from Python repositories, designed to test an agent's ability to navigate codebases and write functional patches.

**5. What does Cohen's Kappa measure in the context of LLM-as-a-judge?**
It measures the inter-rater agreement between the LLM judge and human evaluators (or another LLM), accounting for the probability of them agreeing by chance.

**6. Define Trajectory Efficiency.**
It is the ratio of the reward (success) to the cost incurred (length of trajectory multiplied by token/compute cost).

**7. What is position bias in LLM evaluators?**
The tendency of an LLM to disproportionately favor the first (or sometimes the last) option presented to it when comparing two models' outputs.

**8. How can position bias be mitigated?**
By evaluating the outputs twice, swapping their order in the prompt, and only accepting the result if the LLM prefers the same model in both positions.

**9. What is GAIA?**
A benchmark that tests general AI assistants on tasks requiring reasoning, tool use, and web browsing.

**10. Why do we need sandboxed environments for agent evaluation?**
Because agents take actions that can alter the system state (e.g., executing code, modifying files, making network requests). Sandboxing ensures evaluations are safe and reproducible.

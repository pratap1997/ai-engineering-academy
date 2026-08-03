# Engineering Challenge: Build a CI Evaluation Harness

**Objective:** Build a Continuous Integration (CI) Evaluation Harness for AI Coding Agents.

## Requirements

1. **Test Runner Integration**: Write a script that can take a proposed code patch from an agent and a path to a test file, apply the patch, and run the test file in an isolated environment.
2. **Trajectory Parsing**: The harness must parse the agent's output logs to extract the number of steps taken and the tokens consumed.
3. **Automated Judging**: Implement an LLM-as-a-judge (you can use an API for this part) that reads the trajectory and scores the agent's "reasoning quality" on a 1-5 scale based on a provided rubric.
4. **Report Generation**: Output a markdown report showing Pass@K, average trajectory efficiency, and the judge's feedback.

**Constraints:**
- Do not use pre-built evaluation frameworks like LangChain Evaluators.
- All evaluation logic must be written from scratch.
- The environment isolation can be simulated or implemented using Docker.

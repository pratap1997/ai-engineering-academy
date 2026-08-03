# Mental Model: The Flight Simulator for Agents

Think of agent evaluation like a **Flight Simulator for Pilots**.

When you train a pilot, you don't just ask them multiple-choice questions about aerodynamics (that's static evaluation). You put them in a simulator where they have to take off, navigate turbulence, and land safely. If they crash in the simulator, nobody gets hurt, but you get a detailed log of every dial they turned and every lever they pulled.

## Components of the Simulator

1. **The Scenario (The Task)**: A highly specific problem. For a coding agent, this is a cloned Git repository at a specific commit, an issue description, and a test suite.
2. **The Cockpit (The Tools)**: The actions the agent can take. Search files, read files, run terminal commands, execute Python.
3. **The Black Box (Trajectory Logging)**: Every thought, action, and environment response is recorded. This is the **trajectory**.
4. **The Inspector (The Judge)**: Once the simulation ends, an evaluator checks if the plane landed safely. This can be:
   - **Deterministic Check**: Did the unit tests pass?
   - **LLM-as-a-Judge**: A stronger model (like GPT-4 or Claude 3.5 Sonnet) grades the trajectory based on a rubric (e.g., "Did the agent hallucinate API calls?").

By running thousands of these simulated flights, we can calculate metrics like Pass@K and Trajectory Efficiency.

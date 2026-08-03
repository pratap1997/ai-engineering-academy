# Mathematics of Agent Evaluation

When benchmarking agents, we rely on several key mathematical formulas to quantify their performance.

## 1. Pass@K for Trajectories

Agents are stochastic. Running an agent once on a task might fail, but running it 5 times might yield 1 success. To estimate the probability of at least one success when generating $K$ samples, given a total of $N$ evaluations where $c$ are correct, we use the Pass@K formula:

$$ Pass@K = 1 - \frac{\binom{N-c}{K}}{\binom{N}{K}} $$

This gives an unbiased estimate of the agent's ability to solve the task if allowed $K$ attempts.

## 2. LLM-as-a-Judge Agreement: Cohen's Kappa ($\kappa$)

When using an LLM to judge agent outputs, we want to know how much the LLM agrees with human annotators (or another LLM). We use Cohen's Kappa to measure inter-rater reliability, accounting for chance agreement:

$$ \kappa = \frac{p_o - p_e}{1 - p_e} $$

Where $p_o$ is the relative observed agreement among raters, and $p_e$ is the hypothetical probability of chance agreement.

## 3. Trajectory Efficiency Score ($E$)

Success isn't everything. An agent that solves a task in 5 steps for $0.05 is better than one taking 50 steps for $2.00. We define Trajectory Efficiency as:

$$ E(\tau) = \frac{R(\tau)}{\text{Length}(\tau) \cdot \text{Cost}(\tau)} $$

Where:
- $R(\tau)$ is the binary reward (1 for success, 0 for failure)
- $\text{Length}(\tau)$ is the number of steps in trajectory $\tau$
- $\text{Cost}(\tau)$ is the token cost in USD

## 4. G-Eval Weighted Rubric

When using LLMs as evaluators (G-Eval methodology), we prompt the LLM to output a score (1-5) and compute a weighted average based on the log-probabilities of the output tokens representing the numbers 1 to 5:

$$ Score = \sum_{i=1}^5 i \cdot P(y=i | \text{prompt}) $$

Since we don't always have access to logprobs in this from-scratch implementation, our `LLMJudgeEvaluator` will use structural parsing of the model's text response, optionally with bias correction.

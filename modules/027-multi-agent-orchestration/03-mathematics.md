# Mathematical Foundations of Multi-Agent Orchestration

## Formal Definition
A multi-agent system (MAS) can be formally defined as a tuple:
$$ M = (A, E, R, O, T) $$

Where:
* $A = \{a_1, a_2, ..., a_n\}$ is a set of agents.
* $E$ is the environment (or shared state/blackboard).
* $R = \{r_1, r_2, ..., r_n\}$ is the set of reward functions for each agent. In cooperative systems, $r_1 = r_2 = ... = r_n$.
* $O$ represents the observations each agent can make.
* $T$ represents the transition function of the environment state based on joint actions.

## Message Passing Algebra
The state of an agent $a_i$ at time $t+1$ is a function of its current state and the messages it receives. Let $m_i^t$ be the set of incoming messages for agent $a_i$ at time $t$. The state update is given by:

$$ s_i^{(t+1)} = f(s_i^t, m_i^t) $$

Where $f$ represents the agent's internal reasoning function (often parameterized by an LLM).

## Game Theory and Nash Equilibrium
In competitive multi-agent systems (e.g., debate models or red-teaming), agents may have adversarial reward functions. A Nash Equilibrium is a state where no agent $a_i$ can increase its expected reward by changing its strategy, assuming the strategies of all other agents remain constant.

If $S_i$ is the strategy space of agent $i$, a strategy profile $(s_1^*, ..., s_n^*)$ is a Nash Equilibrium if:
$$ \forall i, \forall s_i \in S_i : R_i(s_i^*, s_{-i}^*) \geq R_i(s_i, s_{-i}^*) $$

## Condorcet Voting for Consensus
When an ensemble of agents must reach a consensus, voting mechanisms are employed. The Condorcet method is a ranked-choice voting system that elects the candidate that would win a majority of the vote in all head-to-head elections against all other candidates.

For a set of options $O$, a Condorcet winner $c \in O$ exists if:
$$ \forall x \in O \setminus \{c\} : |\{a \in A \mid P_a(c) > P_a(x)\}| > |\{a \in A \mid P_a(x) > P_a(c)\}| $$
Where $P_a(x) > P_a(y)$ denotes that agent $a$ prefers $x$ over $y$. Note that due to Condorcet's paradox, a winner is not mathematically guaranteed to exist.

## Communication Complexity
The topology of the orchestration dictates the communication overhead (message complexity).

* **Mesh (All-to-All)**: Every agent broadcasts to every other agent. For $n$ agents, the number of messages per round is $O(n^2)$.
* **Star Topology**: All communication routes through a central supervisor. The supervisor communicates with $n-1$ workers, yielding a complexity of $O(n)$.

## Termination Conditions
A multi-agent sequence must have defined termination conditions to prevent infinite loops:
1. **Fixed-point convergence**: State $E_{t+1} = E_t$, meaning no further messages alter the blackboard.
2. **Quorum**: A predefined majority (e.g., 2/3 of agents) agree on the final state.
3. **Timeout**: An absolute limit on steps or tokens consumed.

# Mathematics of Tree Search

The cornerstone of Monte Carlo Tree Search is balancing exploration (trying new things) with exploitation (following known good paths). The UCB1 (Upper Confidence Bound) algorithm provides the Upper Confidence bounds applied to Trees (UCT) formula.

$$UCT(v_i) = \frac{Q(v_i)}{N(v_i)} + c \sqrt{\frac{\ln N(v_{parent})}{N(v_i)}}$$

Where:
- $Q(v_i)$ is the total reward accumulated by node $v_i$.
- $N(v_i)$ is the number of times node $v_i$ was visited.
- $N(v_{parent})$ is the number of times the parent node was visited.
- $c$ is the exploration constant (often $\sqrt{2}$).

### Value Update Rule
During backpropagation, the value estimate is updated:
$$Q(v) \leftarrow Q(v) + \frac{r - Q(v)}{N(v)}$$

### Complexity
Tree search must grapple with the branching factor $b$ and depth $d$, leading to a time/space complexity of $O(b^d)$ in naive implementations, mitigated by the guided nature of MCTS.

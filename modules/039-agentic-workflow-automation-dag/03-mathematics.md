# 03 - Mathematics: Graph Theory and Execution Formalism

## Directed Acyclic Graph (DAG)
A workflow is represented as a DAG $G = (V, E)$, where:
- $V$ is the set of vertices (workflow nodes/tasks).
- $E$ is the set of directed edges (dependencies). An edge $(u, v) \in E$ means $u$ must execute before $v$.

## Topological Sorting (Kahn's Algorithm)
To execute the DAG, we must find a linear ordering of $V$ such that for every directed edge $(u, v)$, $u$ comes before $v$. 
Kahn's algorithm repeatedly removes vertices with in-degree $0$:
1. $L \leftarrow$ Empty list that will contain the sorted elements.
2. $S \leftarrow$ Set of all nodes with no incoming edge.
3. While $S$ is non-empty:
   - Remove a node $n$ from $S$ and insert it into $L$.
   - For each node $m$ with an edge $e$ from $n$ to $m$:
     - Remove edge $e$ from the graph.
     - If $m$ has no other incoming edges, insert $m$ into $S$.
4. If graph has edges, a cycle exists.

## State Transition Function
Let $S$ be the state space. Each node $v \in V$ defines a transition function:
$$ \delta_v: S \times A \to S $$
Where $A$ is the action space of the agent/task. The overall execution is the composition of these functions along the topological order.

## Exponential Backoff Delay
When an error occurs, retry delays follow exponential backoff to prevent thundering herd problems:
$$ t_{retry} = \min(t_{max}, t_0 \cdot 2^{k-1} + \text{jitter}) $$
Where $k$ is the retry attempt number, $t_0$ is the base delay, and $t_{max}$ is the maximum allowed delay.

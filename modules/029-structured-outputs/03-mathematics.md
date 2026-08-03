# 03 - Mathematics of Structured Outputs

## JSON Schema as a Formal Grammar
A JSON schema can be represented as a Context-Free Grammar (CFG), $G = (V, \Sigma, R, S)$, where:
- $V$ is a set of non-terminal symbols (e.g., `<String>`, `<Number>`, `<Object>`).
- $\Sigma$ is a set of terminal symbols (the vocabulary of the LLM).
- $R$ is a set of production rules derived from the JSON Schema.
- $S$ is the start symbol (usually an `<Object>`).

For example, a schema requiring an integer `age` translates to a production rule:
$R_{\text{age}} \rightarrow \text{"age": } \langle \text{Integer} \rangle$

## Constrained Decoding via Token Masking
During standard generation, the LLM produces a probability distribution over the vocabulary $V$ for the next token $t_i$:
$P(t_i | t_{<i})$

In **constrained decoding**, we define an indicator function based on the grammar $G$:
$\text{valid}(t_i, t_{<i}, G) = \begin{cases} 1 & \text{if concatenating } t_i \text{ to } t_{<i} \text{ is a valid prefix in } G \\ 0 & \text{otherwise} \end{cases}$

The constrained probability distribution becomes:
$P^*(t_i | t_{<i}, G) = \frac{P(t_i | t_{<i}) \cdot \text{valid}(t_i, t_{<i}, G)}{\sum_{t \in V} P(t | t_{<i}) \cdot \text{valid}(t, t_{<i}, G)}$

This effectively masks out any tokens that would violate the JSON schema, forcing the probability to 0 and redistributing the remaining probability mass.

## JSON Schema Validation as a Pushdown Automaton
A pushdown automaton (PDA) is a finite automaton equipped with a stack. Validating a JSON structure requires a PDA because of nested structures (objects within arrays within objects).
The PDA reads the generated JSON character by character. When it encounters `{` or `[`, it pushes onto the stack. When it encounters `}` or `]`, it pops. The string is structurally valid if the stack is empty at the end.

## Information Theory: Schema Constraints Reduce Entropy
The entropy $H$ of the LLM's next-token distribution is:
$H(T_i) = - \sum_{t \in V} P(t) \log_2 P(t)$

When we apply a schema constraint $G$, we drastically reduce the vocabulary of valid tokens to a subset $V_{valid} \subset V$. This reduces the conditional entropy $H(T_i | G) < H(T_i)$, meaning we have removed bits of uncertainty from the model's generation process.

## Tool Selection as Classification
When an LLM is presented with $K$ possible tools, tool selection can be modeled as a classification problem over the tool descriptions.
Given context $C$ and tool descriptions $D_1, ..., D_K$:
$P(\text{Tool}_k | C) \approx \text{Softmax}(\text{Score}(C, D_k))$

The LLM computes the relevance of the context to each tool description and selects the $\text{argmax}_k P(\text{Tool}_k | C)$. Once selected, it transitions into constrained generation for that tool's specific arguments schema.

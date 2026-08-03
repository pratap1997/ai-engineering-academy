# Module 002: Real Applications — Feature Learning & Multi-Class Systems

## 1. Production Use Cases for Multilayer Feedforward Networks

While Module 002 focuses on forward pass mechanics, multi-layer feedforward networks form the backbone of modern deep learning applications:

### Case 1: Automatic Feature Extraction (Representation Learning)
In traditional machine learning, engineers spent months manually hand-crafting features (e.g., polynomial combinations, ratios, domain-specific indicators).
In an MLP:
- Layer 1 learns low-level combination primitives (e.g., AND/OR/NAND boundary lines).
- Layer 2 combines primitives into high-level geometric regions (polygons, enclosures).
- Layer 3 makes the final linear classification on the disentangled representation.

### Case 2: Multi-Class Classification (Softmax Layer)
For single-class binary problems, output $y \in \{0, 1\}$.
For multi-class problems (e.g., classifying images into 10 digits $0-9$):
- Output layer has $K=10$ neurons.
- Activation function: **Softmax**:
  $$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
- Transforms raw logits $z_i$ into a valid probability distribution where $\sum_{i=1}^K P(y=i) = 1$.

### Case 3: Tabular Data Risk & Credit Scoring
Financial institutions use multi-layer feedforward architectures on tabular customer records:
- Input: Age, Income, Debt-to-Income ratio, Credit utilization, Delinquency history.
- Hidden Layers: Learn non-linear risk interaction terms (e.g., "high income but recent delinquency").
- Output: Probability of default $P(\text{Default} | \mathbf{x}) \in [0, 1]$.

---

## 2. Engineering Considerations

1. **Depth vs Width**:
   - **Width** (more neurons per layer): Increases memory and parallel capacity to represent fine-grained boundaries in 2D/3D space.
   - **Depth** (more layers stacked): Exponentially increases representational capability for structured/hierarchical problems with fewer parameters.

2. **Activation Selection**:
   - Hidden Layers: **ReLU** is default due to speed and avoidance of vanishing gradients.
   - Output Layer:
     - Binary Classification: **Sigmoid** or **Step**.
     - Multi-Class Classification: **Softmax**.
     - Regression: **Linear** (no activation).

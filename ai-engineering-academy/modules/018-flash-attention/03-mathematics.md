# Module 018: Mathematics — Online Softmax Recurrence & FlashAttention Algorithm

## 1. Online Softmax Recurrence Relations

Given vector $\mathbf{x} = [\mathbf{x}^{(1)}, \mathbf{x}^{(2)}]$ split into two blocks $\mathbf{x}^{(1)}$ and $\mathbf{x}^{(2)}$.

Let statistics for block 1 be:
$$m^{(1)} = \max_{j} x_j^{(1)}, \quad d^{(1)} = \sum_{j} e^{x_j^{(1)} - m^{(1)}}$$

Let statistics for block 2 be:
$$m^{(2)} = \max_{j} x_j^{(2)}, \quad d^{(2)} = \sum_{j} e^{x_j^{(2)} - m^{(2)}}$$

The **combined online statistics** $(m^\text{new}, d^\text{new})$ are:

$$m^\text{new} = \max(m^{(1)}, m^{(2)})$$
$$d^\text{new} = e^{m^{(1)} - m^\text{new}} \cdot d^{(1)} + e^{m^{(2)} - m^\text{new}} \cdot d^{(2)}$$

### Output Accumulator Recurrence Formula
If $\mathbf{O}^{(1)}$ is the partial output vector for block 1:

$$\mathbf{O}^\text{new} = \text{diag}\left(\frac{e^{m^{(1)} - m^\text{new}} \cdot d^{(1)}}{d^\text{new}}\right) \mathbf{O}^{(1)} + \text{diag}\left(\frac{e^{m^{(2)} - m^\text{new}}}{d^\text{new}}\right) e^{\mathbf{x}^{(2)} - m^{(2)}} \mathbf{V}^{(2)}$$

---

## 2. FlashAttention-1 Tiled Algorithm Pseudo-Code

Given Query $\mathbf{Q} \in \mathbb{R}^{T \times d}$, Key $\mathbf{K} \in \mathbb{R}^{T \times d}$, Value $\mathbf{V} \in \mathbb{R}^{T \times d}$, tile sizes $B_r, B_c$:

1. Partition $\mathbf{Q}$ into $T_r = \lceil T / B_r \rceil$ blocks of size $B_r \times d$.
2. Partition $\mathbf{K}, \mathbf{V}$ into $T_c = \lceil T / B_c \rceil$ blocks of size $B_c \times d$.
3. Initialize Output $\mathbf{O} = \mathbf{0}_{T \times d}$, row-max vector $\mathbf{m} = -\boldsymbol{\infty}_T$, sum vector $\mathbf{d} = \mathbf{0}_T$ in HBM.

**Outer Loop** over Key/Value blocks $j = 1 \dots T_c$:
  - Load Key block $\mathbf{K}_j$ and Value block $\mathbf{V}_j$ into SRAM.
  - **Inner Loop** over Query blocks $i = 1 \dots T_r$:
    - Load Query block $\mathbf{Q}_i$, partial output $\mathbf{O}_i$, local stats $\mathbf{m}_i, \mathbf{d}_i$ into SRAM.
    - Compute block scores: $\mathbf{S}_{ij} = \frac{\mathbf{Q}_i \mathbf{K}_j^T}{\sqrt{d}} \in \mathbb{R}^{B_r \times B_c}$.
    - Compute local max: $\tilde{\mathbf{m}}_{ij} = \text{rowmax}(\mathbf{S}_{ij}) \in \mathbb{R}^{B_r}$.
    - Compute local exponentiated scores: $\tilde{\mathbf{P}}_{ij} = \exp(\mathbf{S}_{ij} - \tilde{\mathbf{m}}_{ij})$.
    - Compute local sum: $\tilde{\mathbf{d}}_{ij} = \text{rowsum}(\tilde{\mathbf{P}}_{ij}) \in \mathbb{R}^{B_r}$.
    - Update new max: $\mathbf{m}_i^\text{new} = \max(\mathbf{m}_i, \tilde{\mathbf{m}}_{ij})$.
    - Update new sum: $\mathbf{d}_i^\text{new} = e^{\mathbf{m}_i - \mathbf{m}_i^\text{new}} \odot \mathbf{d}_i + e^{\tilde{\mathbf{m}}_{ij} - \mathbf{m}_i^\text{new}} \odot \tilde{\mathbf{d}}_{ij}$.
    - Update output accumulator:
      $$\mathbf{O}_i \leftarrow \text{diag}\left(\frac{\mathbf{d}_i e^{\mathbf{m}_i - \mathbf{m}_i^\text{new}}}{\mathbf{d}_i^\text{new}}\right) \mathbf{O}_i + \text{diag}\left(\frac{e^{\tilde{\mathbf{m}}_{ij} - \mathbf{m}_i^\text{new}}}{\mathbf{d}_i^\text{new}}\right) \tilde{\mathbf{P}}_{ij} \mathbf{V}_j$$
    - Write updated $\mathbf{O}_i, \mathbf{m}_i^\text{new}, \mathbf{d}_i^\text{new}$ back to HBM.

Returns final Output $\mathbf{O} \in \mathbb{R}^{T \times d}$.

# Mathematics of Layout Analysis

In layout-aware document processing, we must quantitatively analyze the spatial relationships between bounding boxes.

### Spatial Intersection over Union (IoU)

To determine if two detected text blocks belong to the same logical unit (or if a detection matches a ground-truth annotation), we use the Spatial IoU. Let $B_1$ and $B_2$ represent the rectangular bounding boxes of two elements.

$$
\text{IoU} = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)}
$$

An IoU of 0 means the boxes are completely disjoint, while an IoU of 1 means they perfectly overlap.

### Table Alignment Score

When reconstructing tables, we evaluate the vertical and horizontal alignment of cells. For two cells $C_i$ and $C_j$ in the same column, their horizontal alignment score can be measured by the deviation of their center X-coordinates or left edges. 

Let $x_{\text{left}}(C)$ be the left x-coordinate of cell $C$. A simple alignment penalty $P$ is:
$$ P(C_i, C_j) = |x_{\text{left}}(C_i) - x_{\text{left}}(C_j)| $$
A lower penalty indicates better column alignment.

### Layout Entropy

To measure the complexity of a document's layout, we can borrow the concept of entropy from information theory. Let $P(l_i)$ be the probability (or proportional area) of a specific layout class $l_i$ (e.g., text, table, figure, header) appearing on a page. The Layout Entropy $H(L)$ is:

$$
H(L) = -\sum_{i} P(l_i) \log P(l_i)
$$

A page of pure text has low entropy, while a complex dashboard with mixed media has high entropy, indicating the need for sophisticated parsing strategies.

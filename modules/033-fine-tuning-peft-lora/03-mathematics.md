# Mathematics of LoRA
The weight update is given by: $ W' = W_0 + \frac{\alpha}{r} (B \cdot A) $
- $ W_0 \in \mathbb{R}^{d_{out} \times d_{in}} $
- $ B \in \mathbb{R}^{d_{out} \times r} $
- $ A \in \mathbb{R}^{r \times d_{in}} $
Parameter count savings: instead of $ d_{in} \cdot d_{out} $ parameters, we only train $ r \cdot (d_{in} + d_{out}) $. Since $ r \ll \min(d_{in}, d_{out}) $, this is a massive reduction.

# 08 - Assessment: Long-Context Scaling

1. **Why does standard attention scale poorly to long contexts?**
   - *Answer*: Because the attention matrix size grows quadratically ($O(N^2)$) with the sequence length $N$, leading to unacceptable memory and compute costs.

2. **What happens to standard RoPE embeddings when tested on lengths beyond the training distribution?**
   - *Answer*: The model encounters high-frequency rotations and relative token distances it has never learned, causing performance to drop significantly.

3. **How does Linear RoPE Scaling solve the out-of-distribution problem?**
   - *Answer*: It divides the position indices by a scaling factor, mathematically squishing the longer sequence into the original training length.

4. **What is the primary drawback of Linear RoPE Scaling?**
   - *Answer*: It compresses all frequencies equally, which blurs local context (high frequencies) that the model already knows how to handle well.

5. **How does NTK-aware scaling improve upon Linear Scaling?**
   - *Answer*: It changes the base of the RoPE frequencies, preserving high frequencies (local relationships) while aggressively scaling low frequencies (global relationships).

6. **In the YaRN method, what is the purpose of the ramp function?**
   - *Answer*: The ramp function smoothly transitions the interpolation between the fast-moving, untouched dimensions and the slow-moving, heavily scaled dimensions.

7. **What is Temperature Scaling in YaRN and why is it used?**
   - *Answer*: It multiplies the attention logits by a scalar $> 1$. It compensates for the loss of sharpness in attention scores caused by squishing tokens together.

8. **How does Sliding Window Attention (SWA) reduce memory usage?**
   - *Answer*: By only computing attention for the most recent $W$ tokens instead of all $N$ previous tokens, changing complexity from $O(N^2)$ to $O(N \times W)$.

9. **In SWA, how can a model access information outside its immediate window?**
   - *Answer*: Through deep layers; information from token A can pass to token B in layer 1, and then from token B to token C in layer 2.

10. **What is Chunked Prefill and what problem does it solve?**
    - *Answer*: Processing a long prompt in smaller blocks rather than all at once. It prevents memory spikes and out-of-memory errors during the initial KV-cache computation.

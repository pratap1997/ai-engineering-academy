# Module 019: Mental Model — The Translucent Overlay Sheet

## 1. The Translucent Plastic Overlay Analogy

Imagine a huge printed textbook (The 70B Base Model):

- **Full Fine-Tuning**: Erasing every printed page in the textbook and re-printing it with new ink. High cost, destroys original knowledge, requires a massive printing press (huge VRAM).
- **LoRA**: Placing a tiny, transparent plastic overlay sheet over specific pages. You write only small annotations on the overlay sheet. The original printed textbook is untouched (frozen).
- **Inference Merging**: Printing the final updated textbook by flattening the overlay annotations directly onto the base pages ($\mathbf{W}_\text{merged} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}$).

---

## 2. Why Low-Rank Works (Intrinsic Dimensionality)

Research by Aghajanyan et al. (2020) showed that pre-trained language models have a very low **intrinsic dimension**.

Even though a weight matrix $\mathbf{W}_0$ is $4096 \times 4096$ ($16.7\text{M}$ numbers), the actual weight updates $\Delta \mathbf{W}$ during fine-tuning lie in a tiny sub-space:

$$\text{Rank}(\Delta \mathbf{W}) \le r \quad (r = 4, 8, \text{ or } 16)$$

By factoring $\Delta \mathbf{W} \in \mathbb{R}^{4096 \times 4096}$ into two narrow matrices:
- $\mathbf{A} \in \mathbb{R}^{8 \times 4096}$ ($32,768$ params)
- $\mathbf{B} \in \mathbb{R}^{4096 \times 8}$ ($32,768$ params)

Total trainable params = $65,536$ ($99.6\%$ reduction!).

# Module 018: Mental Model — The Conveyor Belt & SRAM Tiles

## 1. The Conveyor Belt & Workbench Analogy

Imagine a factory worker (GPU Tensor Core) processing a huge document:

- **Standard Attention**: The worker writes out a massive $10,000 \times 10,000$ grid of similarity scores on paper, walks across the factory floor to file it in the warehouse (HBM), walks back to fetch it, divides every row by the sum, walks back to the warehouse, and finally multiplies by the values. The worker spends **90% of their time walking back and forth to the warehouse (Memory IO)**!
- **FlashAttention**: The worker keeps a tiny workbench (SRAM cache). They process 64 lines of text at a time on their workbench, keep a running running-maximum and running-sum in a tiny pocket notebook (Online Softmax), and write only the final 64 output vectors back to the warehouse.

```
GPU Memory Hierarchy:
┌─────────────────────────────────────────────────────────┐
│ SRAM (Fast Cache): 100-256 KB per SM | ~19 TB/s Bandwidth │  <-- FlashAttention operates here!
└─────────────────────────────────────────────────────────┘
                            ▲
                            │ Block Tile Transfers
                            ▼
┌─────────────────────────────────────────────────────────┐
│ HBM (Main VRAM):   24-80 GB           | ~2-3 TB/s Bandwidth│  <-- Standard Attention spills here!
└─────────────────────────────────────────────────────────┘
```

---

## 2. Online Softmax Core Intuition

Standard Softmax over vector $\mathbf{x} = [x_1, x_2, \dots, x_N]$ requires 2 full passes:
1. Pass 1: Find global maximum $m = \max(x_1, \dots, x_N)$ and sum $d = \sum e^{x_i - m}$.
2. Pass 2: Divide each $e^{x_i - m}$ by $d$.

**Online Softmax** updates $m$ and $d$ dynamically when receiving a new block of data $\mathbf{x}^{(2)}$:

If old max is $m^{(1)}$ and new block max is $m^{(2)}$, the new combined max is:
$$m^\text{new} = \max(m^{(1)}, m^{(2)})$$

The previous sum $d^{(1)}$ is rescaled by $e^{m^{(1)} - m^\text{new}}$ so it matches the new baseline!
No need to re-read previous elements from HBM!

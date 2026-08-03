# Module 009: Advanced Recurrent Architectures (LSTM, GRU & Gated Memory Units)

> "Vanilla RNNs suffer from amnesia after 10–20 time steps due to vanishing gradients. LSTMs and GRUs introduce explicit gating mechanisms that allow the model to selectively remember, forget, or write information across hundreds of temporal steps."

---

## 1. Motivation: Long-Term Dependencies & The Memory Bottleneck

In **Module 008**, we saw how unrolling backpropagation across long sequences causes gradients to explode or vanish.
While **Gradient Clipping** caps exploding gradients, it cannot fix **vanishing gradients**:
- If a key piece of information is presented at step $t=1$ (e.g. *"Alice was born in France..."*) and the model must answer a question at step $t=100$ (*"...she speaks fluent ____"*), a Vanilla RNN has completely erased $\mathbf{h}_1$ from its memory state $\mathbf{h}_{100}$.

**Module 009** introduces the two foundational architectures that solved long-term temporal dependencies:
1. **Long Short-Term Memory (LSTM)** (Hochreiter & Schmidhuber, 1997): Introduces an additive **Cell State ($\mathbf{C}_t$)** acting as a constant error carousel, guarded by 3 sigmoid gates (Forget $\mathbf{f}_t$, Input $\mathbf{i}_t$, Output $\mathbf{o}_t$).
2. **Gated Recurrent Unit (GRU)** (Cho et al., 2014): Streamlines LSTM by coupling Forget and Input gates into a single **Update Gate ($\mathbf{z}_t$)** and **Reset Gate ($\mathbf{r}_t$)**.

---

## 2. Learning Outcomes

By completing this module, you will be able to:

1. **Implement LSTM & GRU Primitives**: Build `LSTMCell`, `LSTMSequence`, `GRUCell`, and `GRUSequence` from scratch in pure Python & NumPy.
2. **Derive the Constant Error Carousel (CEC)**: Prove mathematically why the additive cell state update $\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \mathbf{\tilde{C}}_t$ preserves long-term memory without vanishing gradients.
3. **Compare LSTM vs GRU**: Analyze parameter efficiency ($4 \times$ weights vs $3 \times$ weights) and convergence speeds across sequence tasks.
4. **Build a Gated Language Model**: Construct an LSTM-based sequence model for character and word prediction.

---

## 3. Module Roadmap

```
01-overview.md          → You are here
02-mental-model.md      → The conveyor belt analogy (Cell State C_t) & Gated write/erase/read operations
03-mathematics.md       → Formal equations of LSTM (4 gates) & GRU (2 gates), CEC derivative proof, parameter counts
04-implementation.py    → Pure Python & NumPy implementations of LSTMCell, LSTMSequence, GRUCell, GRUSequence
05-experiments.py       → 100-step long-term dependency retrieval benchmark (Vanilla RNN vs LSTM) & GRU speed check
06-real-applications.md → PyTorch torch.nn.LSTM & torch.nn.GRU, Bidirectional & Stacked LSTMs, Speech & Financial models
07-engineering-challenge.md → Custom LSTMSequence with unrolled BPTT backward pass & gradcheck verification
08-assessment.md        → Readiness check & self-assessment rubrics
09-references.md        → Hochreiter & Schmidhuber (1997) & Cho et al. (2014) citations
```

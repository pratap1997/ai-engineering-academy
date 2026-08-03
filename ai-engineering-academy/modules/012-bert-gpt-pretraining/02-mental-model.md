# Module 012: Mental Model — Fill-in-the-Blank vs. Next-Word Prediction

## 1. BERT: The Fill-in-the-Blank Game

BERT pre-trains by randomly masking 15% of tokens and asking the model to predict them:

```
Input:   "The [MASK] sat on the [MASK]"
Target:  "The  cat  sat on the  mat"
```

Because BERT sees **both left and right context** (bidirectional), it can use "sat on the" to predict "cat" AND "The cat sat on the" to predict "mat".

**The 80/10/10 masking rule** (prevents the model from learning that `[MASK]` tokens always need prediction):
- **80%** of chosen positions → replace with `[MASK]`
- **10%** of chosen positions → replace with a random token
- **10%** of chosen positions → keep the original token

Only positions that were chosen (15% of tokens) contribute to the loss.

---

## 2. GPT: The Next-Word Prediction Game

GPT pre-trains by predicting the next token given all previous tokens:

```
Input:   "The cat sat on the"
Target:  "cat sat on the mat"
```

GPT uses **causal (autoregressive) attention** — each position can only attend to positions $\leq t$. This makes it naturally generative: at inference, generate one token → append → generate the next.

---

## 3. Why Do They Learn So Much From This?

Both tasks force the model to build a **compressed world model** inside its weights:
- To predict `[MASK]` in "The [MASK] barked at the mailman", the model must know that "cats meow" and "dogs bark".
- To predict the next word in "The Eiffel Tower is in [NEXT]", the model must store geographic facts.

The representations that emerge — BERT's `[CLS]` token embedding or GPT's final hidden state — encode syntax, semantics, facts, and reasoning in a dense vector that can be fine-tuned for any downstream task.

# Module 001 — Perceptron From Scratch

**Status:** `draft` | **Difficulty:** Beginner | **Estimated time:** 4 hours

---

## Capability Contract

By the end of this module you will be able to:

- ✓ Explain how a perceptron converts weighted inputs into a binary decision
- ✓ Derive the prediction and update equations from first principles
- ✓ Implement it using plain Python (no ML libraries)
- ✓ Implement an equivalent NumPy version
- ✓ Train it on AND and OR gates and verify with tests
- ✓ Diagnose and explain why it fails on XOR (geometrically, not just empirically)
- ✓ Complete a constrained engineering challenge without copying the reference implementation
- ✓ Explain when a perceptron should **not** be used

---

## Learning Path

Work through the artifacts in order. Do not skip ahead to the challenge.

| Step | Artifact | Purpose |
|---|---|---|
| 1 | [01-overview.md](./01-overview.md) | Understand the problem and context |
| 2 | [02-mental-model.md](./02-mental-model.md) | Build intuition before equations |
| 3 | [03-mathematics.md](./03-mathematics.md) | Derive the rules formally |
| 4 | [04-implementation.py](./04-implementation.py) | Read the reference implementation |
| 5 | [05-experiments.ipynb](./05-experiments.ipynb) | Run and observe experiments |
| 6 | [06-real-applications.md](./06-real-applications.md) | Understand where it is used and when not |
| → | **Readiness Check** | 5 questions at the end of step 6 |
| 7 | [07-engineering-challenge.md](./07-engineering-challenge.md) | Build it yourself |
| 8 | [08-assessment.md](./08-assessment.md) | Check understanding, see answers |
| 9 | [09-references.md](./09-references.md) | Papers, sources, further reading |

---

## Run the Tests

```bash
cd ai-engineering-academy
pip install -r requirements.txt
pytest modules/001-perceptron/tests/ -v
```

All 13 tests must pass against the reference implementation before you attempt the challenge.

---

## Scope of This Module

**Included:**
- Single-layer binary perceptron
- 0/1 label convention
- AND and OR learning
- XOR failure analysis
- Decision boundary visualization
- Plain Python and NumPy implementations

**Not included (later modules):**
- Multilayer networks
- Backpropagation
- Loss functions and gradient descent
- PyTorch
- Multiclass classification
- Production deployment

---

## Feedback

Something unclear? Find an error? Complete the challenge?

→ [Open a GitHub Issue](../../.github/ISSUE_TEMPLATE/)  
→ See [CONTRIBUTING.md](../../CONTRIBUTING.md) for submission instructions

---

## Source Material

This module uses the Microsoft AI For Beginners perceptron lesson as a concept seed.
The module structure, experiments, challenge, and assessment are original.

Full provenance: [module.yaml](./module.yaml) → sources section  
Licence governance: [SOURCES.md](../../SOURCES.md)

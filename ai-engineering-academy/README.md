# AI Engineering Academy

> **Learn it. Build it. Teach it. Prove it.**

An open AI engineering curriculum organized around **demonstrable capability**, not content volume.

---

## What Makes This Different

Most AI courses optimize for:
- Watching
- API familiarity
- Completion certificates
- Broad, shallow coverage

This academy optimizes for:

```
Read authoritative source
→ Form a mental model
→ Implement from scratch
→ Test
→ Experiment with controlled variables
→ Discover misconceptions
→ Correct them
→ Teach externally
→ Observe learner confusion
→ Revise the module
```

A module is not complete when you read it.  
It is complete when you can **explain it, implement it, debug it, and prove it**.

---

## Four Permanent Assets

| Asset | Purpose |
|---|---|
| **Knowledge** | Explain what exists and why it works |
| **Skills** | Implement, debug, evaluate |
| **Projects** | Combine concepts into useful systems |
| **Engineering Judgment** | Make decisions, trade off, and document why |

Books, videos, quizzes, and AI tutors are delivery outputs. These four are the product.

---

## Curriculum

### Foundations

| Module | Topic | Status | Estimated Time |
|---|---|---|---|
| [001](./modules/001-perceptron/) | Perceptron From Scratch | `draft` | 4 hours |
| 002 | Loss Functions | `planned` | — |
| 003 | Gradient Descent | `planned` | — |
| 004 | NumPy Neural Network | `planned` | — |
| 005 | Evaluation and Debugging | `planned` | — |

### Mini-Capstone
| Project | Modules Used | Status |
|---|---|---|
| Binary Classifier From Raw Data | 001–005 | `planned` |

---

## How to Use This Academy

### As a learner

1. Open any module's `README.md` — it is your entry point.
2. Read the **capability contract** first. That is what you are building toward.
3. Work through the artifacts in order.
4. Attempt the **engineering challenge** before reading the assessment answers.
5. Run the tests against your own implementation.
6. If something is unclear, open a GitHub issue — that feedback directly improves the module.

### As a contributor

See [CONTRIBUTING.md](./CONTRIBUTING.md) for how to report errors, reproduce experiments, submit challenge attempts, or propose improvements.

---

## Module Standard

Every module contains nine artifacts:

```
01-overview.md             What problem does this solve? When not to use it?
02-mental-model.md         Intuitive explanation and visual representation
03-mathematics.md          Equations, derivations, assumptions
04-implementation.py       Plain Python + NumPy implementations, both readable
05-experiments.ipynb       Controlled, reproducible investigations
06-real-applications.md    Where it appears in real systems
07-engineering-challenge.md Constrained implementation without scaffolding
08-assessment.md           Concept + debugging + communication checks
09-references.md           Sources, licences, further reading
```

---

## Tech Stack

```
Python 3.12     Language
NumPy           Numerical computing
pytest          Capability verification
Jupyter         Interactive experiments
Jupytext        Readable notebook diffs
GitHub Actions  Independent CI verification
YAML            Module metadata
```

---

## Source Material

This academy uses the following as **reference sources** (not copied curriculum):

- **Microsoft AI For Beginners** — MIT licence — primary curriculum seed
- **Dive into Deep Learning** — CC BY-SA 4.0 (book) — pedagogical reference
- **Sebastian Raschka's repositories** — various licences — implementation reference

All sources are tracked with exact commits, access dates, and licence types in each module's `module.yaml` and in [SOURCES.md](./SOURCES.md).

---

## Status Indicators

| Badge | Meaning |
|---|---|
| `draft` | In progress, not yet reviewed |
| `review` | Awaiting external learner attempt |
| `canonical` | All quality gates passed |
| `planned` | Defined, not yet started |

---

## License

This academy's original content is released under MIT.  
Each module documents the licence of every source it consulted.  
See [SOURCES.md](./SOURCES.md) for full provenance.

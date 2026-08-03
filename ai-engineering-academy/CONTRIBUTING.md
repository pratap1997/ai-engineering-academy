# Contributing to the AI Engineering Academy

Thank you for engaging with this curriculum. Your feedback directly improves
the quality of learning for every future learner.

This document explains every way you can contribute.

---

## Table of Contents

- [Report a Content Error](#report-a-content-error)
- [Report a Confusing Explanation](#report-a-confusing-explanation)
- [Report an Experiment Reproduction Failure](#report-an-experiment-reproduction-failure)
- [Submit a Challenge Attempt](#submit-a-challenge-attempt)
- [Propose an Experiment](#propose-an-experiment)
- [Request the Next Module](#request-the-next-module)
- [Pull Request Requirements](#pull-request-requirements)
- [Attribution and Licence Expectations](#attribution-and-licence-expectations)

---

## Report a Content Error

Use this when you find a factual, mathematical, code, or teaching mistake.

**GitHub Issue template:** `content-error`

Include:
- Which artifact is affected (e.g., `02-mental-model.md`, `04-implementation.py`)
- What the current text or code says
- Why it is incorrect (with source, derivation, or reproduction)
- What the correction should be

A content error is recorded in the module's `ERRATA.md` and corrected before the module moves to `canonical` status.

---

## Report a Confusing Explanation

Use this when the explanation is not technically wrong, but unclear, ambiguous, or incomplete.

**GitHub Issue template:** `confusing-explanation`

Include:
- Which artifact and which section
- What you understood it to mean
- What you later discovered it actually means
- What wording or diagram would have helped

Confusion reports are the highest-value feedback this academy receives.
They become assessment questions, mental model improvements, and experiment ideas.

---

## Report an Experiment Reproduction Failure

Use this when you cannot reproduce an experiment result with the stated parameters.

**GitHub Issue template:** `reproduction-failure`

Include:
- Which experiment in `05-experiments.ipynb`
- Your environment (OS, Python version, package versions)
- The exact parameters used (seed, learning rate, epochs, dataset)
- Your observed result versus the documented result
- Any error messages

---

## Submit a Challenge Attempt

After completing the engineering challenge in `07-engineering-challenge.md`:

1. Fork the repository.
2. Create a branch: `challenge/001-perceptron-your-username`.
3. Add your implementation to `submissions/001-perceptron/your-username/`.
4. Include:
   - `solution.py` — your implementation
   - `results.txt` — test output from running `pytest`
   - `notes.md` — what you found difficult, what surprised you, what you would do differently
5. Open a Pull Request using the PR template.

Your submission does **not** need to be perfect. Partial attempts with honest notes are valuable.

---

## Propose an Experiment

If you ran an experiment not listed in the module and found a meaningful result:

1. Open a GitHub issue titled: `Experiment proposal: [description]`
2. Include:
   - Hypothesis
   - Setup (parameters, data, seed)
   - Observation
   - Why this is educationally valuable
   - Reproducibility notes

Strong proposals become official experiments in the next module revision.

---

## Request the Next Module

If you completed Module 001 and want to know when Module 002 is available:

1. Open a GitHub issue titled: `Module request: 002-loss-functions`
2. Include one sentence about what you are trying to build or understand next.

This is the most important business signal for the academy.
It shows that the curriculum sequence is coherent and that there is demand for continuation.

---

## Pull Request Requirements

Every pull request must include:

- A clear description of what changed and why
- The artifact(s) affected
- Evidence that the change is correct:
  - For code: test output
  - For mathematics: derivation or source citation
  - For explanations: source or reasoning
- Confirmation that all tests still pass (`pytest` output)
- Attribution if any external source was consulted

Pull requests that introduce new content without source attribution will not be merged.

---

## Attribution and Licence Expectations

This academy's original content is MIT licensed.

If your contribution:
- introduces content adapted from another source, you must record that source
  in the module's `09-references.md` and in `SOURCES.md`
- introduces code from an external repository, you must verify that its licence
  is compatible with MIT and attribute it clearly
- introduces prose or diagrams adapted from a CC BY-SA source, you must note
  the share-alike obligation

When in doubt: write it independently, cite the source you learned from.

---

## What Happens to Your Feedback

| Feedback type | Where it goes |
|---|---|
| Content error | `ERRATA.md` → corrected in artifact |
| Confusing explanation | `engineering/progression-log.md` misconceptions field → revised artifact |
| Reproduction failure | Module `ERRATA.md` → reproducibility section updated |
| Challenge attempt | `submissions/` directory → quality signal recorded |
| Experiment proposal | Evaluated for next module revision |
| Module request | Tracked as demand signal |

---

*Thank you. Every confusion report, challenge attempt, and reproduction failure
makes this academy more honest and more useful.*

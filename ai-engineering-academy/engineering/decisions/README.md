# Engineering Decisions

This directory stores engineering decision records for the AI Engineering Academy.

A decision record is created **only** when a genuine engineering trade-off occurs —
not for every implementation choice, and never as speculative documentation.

## When to add a decision record

Create a record when you:
- chose between two reasonable technical alternatives
- rejected a commonly-used approach for a documented reason
- discovered a trade-off through benchmarking or experimentation
- changed an earlier decision and need to explain why

## Format

Create a new file: `decisions/YYYY-MM-DD-short-description.md`

```markdown
# Decision: [Title]

Date: YYYY-MM-DD
Status: decided | revisited | superseded

## Context
What problem required a decision?

## Options considered
What alternatives were evaluated?

## Decision
What was chosen and why?

## Consequences
What does this enable or constrain?

## Evidence
What experiment, benchmark, or source informed this?
```

## Current decisions

*None yet. This directory is created now because the infrastructure costs nothing,
but no real decisions have been required yet.*

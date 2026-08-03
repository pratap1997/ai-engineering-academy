# Module 021: Engineering Challenge — Preference Data Collator & Full DPO Loop

## 1. Challenge Task

Construct a self-contained `DPOAlignmentTrainer` in pure Python & NumPy that:
1. Accepts a preference dataset consisting of $(x_i, y_{w,i}, y_{l,i})$ feature representations.
2. Computes policy and reference log probabilities for chosen and rejected completions.
3. Implements DPO loss calculation with reference model frozen.
4. Trains the policy model over 50 iterations, logging DPO loss, implicit reward margins, and accuracy ($p(\hat{r}_w > \hat{r}_l)$).

---

## 2. Validation Criteria

1. Preference accuracy $p(\hat{r}_w > \hat{r}_l)$ reaches $>90\%$ at convergence.
2. DPO loss decreases monotonically.
3. Reference model parameters remain $100\%$ untouched (frozen).

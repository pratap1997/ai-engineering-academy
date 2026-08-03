# Module 019: Engineering Challenge — Multi-Adapter Dynamic Switching Layer

## 1. Challenge Task

Construct a self-contained `MultiAdapterLoRALinear` layer in pure Python & NumPy that:
1. Wraps a single frozen base linear weight matrix $\mathbf{W}_0$.
2. Stores multiple distinct LoRA adapters (e.g. `adapter_code`, `adapter_math`, `adapter_medical`).
3. Supports dynamic adapter switching during the forward pass: `forward(x, adapter_name="adapter_code")`.
4. Verifies that switching adapters alters output representations correctly while preserving identical base weight representations.

---

## 2. Validation Criteria

1. Dynamic adapter switching evaluates correctly without modifying base weights $\mathbf{W}_0$.
2. `adapter_code` output differs from `adapter_math` output.
3. Merging and unmerging specific adapters maintains mathematical equivalence.

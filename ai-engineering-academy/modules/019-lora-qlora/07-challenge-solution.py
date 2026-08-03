"""
AI ENGINEERING ACADEMY -- MODULE 019 ENGINEERING CHALLENGE SOLUTION
Multi-Adapter Dynamic Switching Layer & Verification
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod19", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

LoRALinear = _mod.LoRALinear


class MultiAdapterLoRALinear:
    """
    Linear layer supporting dynamic switching between multiple LoRA adapters on a single frozen base weight.
    """

    def __init__(self, in_features, out_features, seed=42):
        self.in_features = in_features
        self.out_features = out_features

        np.random.seed(seed)
        scale = np.sqrt(2.0 / in_features)
        self.weight_base = np.random.randn(out_features, in_features) * scale

        self.adapters_A = {}
        self.adapters_B = {}
        self.scalings = {}

    def add_adapter(self, adapter_name, r=8, lora_alpha=16, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.adapters_A[adapter_name] = np.random.randn(r, self.in_features) * (1.0 / np.sqrt(r))
        self.adapters_B[adapter_name] = np.random.randn(self.out_features, r) * 0.01
        self.scalings[adapter_name] = lora_alpha / r

    def forward(self, x, adapter_name=None):
        base_out = np.matmul(x, self.weight_base.T)

        if adapter_name is None or adapter_name not in self.adapters_A:
            return base_out

        A = self.adapters_A[adapter_name]
        B = self.adapters_B[adapter_name]
        scaling = self.scalings[adapter_name]

        lora_out = np.matmul(np.matmul(x, A.T), B.T) * scaling
        return base_out + lora_out


def verify_multi_adapter_switching():
    print("=" * 65)
    print("MODULE 019 CHALLENGE: MULTI-ADAPTER DYNAMIC SWITCHING")
    print("=" * 65)

    np.random.seed(42)
    in_dim, out_dim = 64, 32
    x = np.random.randn(2, 5, in_dim)

    multi = MultiAdapterLoRALinear(in_dim, out_dim, seed=42)

    # Add two specialized adapters
    multi.add_adapter("code_adapter", r=8, lora_alpha=16, seed=101)
    multi.add_adapter("math_adapter", r=8, lora_alpha=16, seed=202)

    out_base = multi.forward(x, adapter_name=None)
    out_code = multi.forward(x, adapter_name="code_adapter")
    out_math = multi.forward(x, adapter_name="math_adapter")

    print(f"Input Shape:      {x.shape}")
    print(f"Base Out Shape:   {out_base.shape}")
    print(f"Code Out Shape:   {out_code.shape}")
    print(f"Math Out Shape:   {out_math.shape}")

    # Verify adapter outputs differ from each other and base
    diff_base_code = np.max(np.abs(out_base - out_code))
    diff_code_math = np.max(np.abs(out_code - out_math))

    print(f"Max Diff (Base vs Code): {diff_base_code:.6f}")
    print(f"Max Diff (Code vs Math): {diff_code_math:.6f}")

    assert diff_base_code > 1e-4
    assert diff_code_math > 1e-4

    print("\nMulti-Adapter Dynamic Switching Verified => [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_multi_adapter_switching()

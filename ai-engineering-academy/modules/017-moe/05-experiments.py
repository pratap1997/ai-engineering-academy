"""
AI ENGINEERING ACADEMY -- MODULE 017 EXPERIMENTS
Expert Utilization Benchmark & MoE Parameter Scaling Analysis
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod17", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

TopKRouter = _mod.TopKRouter
MoELayer   = _mod.MoELayer


def run_experiment_1_expert_utilization():
    print("\n--- EXPERIMENT 1: Expert Token Distribution & Utilization ---")
    np.random.seed(42)
    N, T, d_model = 10, 50, 64
    num_experts, top_k = 8, 2
    x = np.random.randn(N, T, d_model)

    router = TopKRouter(d_model=d_model, num_experts=num_experts, top_k=top_k, seed=42)
    topk_indices, _, aux_loss = router.forward(x)

    flat_indices = topk_indices.flatten()
    counts = np.bincount(flat_indices, minlength=num_experts)

    total_assignments = len(flat_indices)
    expected_per_expert = total_assignments / num_experts

    print(f"  Total Token Assignments: {total_assignments} ({N*T} tokens x top-{top_k})")
    print(f"  Expected per Expert:     {expected_per_expert:.1f}")
    print("  Actual Expert Token Counts:")
    for i, c in enumerate(counts):
        pct = (c / total_assignments) * 100
        print(f"    Expert {i}: {c:4d} tokens ({pct:5.1f}%)")

    # Verify no expert receives 0 tokens (no total collapse)
    assert (counts > 0).all()
    print("  All 8 experts active and utilized [OK]")


def run_experiment_2_moe_parameter_scaling():
    print("\n--- EXPERIMENT 2: Active vs Total Parameter Scaling (Mixtral 8x7B Case Study) ---")
    d_model = 4096
    d_ff = 14336
    num_experts = 8
    top_k = 2

    # Single FFN param count: 2 * d_model * d_ff
    ffn_params = 2 * d_model * d_ff

    dense_ffn_params = ffn_params
    total_moe_ffn_params = num_experts * ffn_params
    active_moe_ffn_params = top_k * ffn_params

    print(f"  Single Expert FFN Params:  {ffn_params / 1e6:6.2f}M")
    print(f"  Total MoE FFN Params:      {total_moe_ffn_params / 1e6:6.2f}M (8 experts)")
    print(f"  Active FFN Params (top-2): {active_moe_ffn_params / 1e6:6.2f}M (2 active)")
    print(f"  Efficiency Factor:         {total_moe_ffn_params / active_moe_ffn_params:6.1f}x total capacity per FLOP!")

    assert total_moe_ffn_params == 4 * active_moe_ffn_params
    print("  MoE parameter capacity scaling verified [OK]")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY -- MODULE 017 EXPERIMENTS")
    print("=" * 70)
    run_experiment_1_expert_utilization()
    run_experiment_2_moe_parameter_scaling()
    print("\n" + "=" * 70)
    print("ALL MODULE 017 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

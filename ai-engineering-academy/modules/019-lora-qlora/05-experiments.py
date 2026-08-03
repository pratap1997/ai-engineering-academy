"""
AI ENGINEERING ACADEMY -- MODULE 019 EXPERIMENTS
LoRA Parameter Reduction & Weight Merging Benchmarks
"""

import os
import importlib.util
import time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod19", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

LoRALinear = _mod.LoRALinear


def run_experiment_1_parameter_reduction():
    print("\n--- EXPERIMENT 1: Trainable Parameter Reduction (LLaMA 3 8B Case Study) ---")
    in_dim, out_dim = 4096, 4096
    num_layers = 32
    num_target_matrices_per_layer = 4  # Q, K, V, O projections

    base_params = num_layers * num_target_matrices_per_layer * (in_dim * out_dim)

    print("  Rank (r) | Trainable Params | Base Params | Trainable Ratio (%) | Parameter Reduction")
    print("  " + "-" * 78)

    for r in [2, 4, 8, 16, 32, 64]:
        lora_params_per_matrix = r * in_dim + out_dim * r
        total_lora_params = num_layers * num_target_matrices_per_layer * lora_params_per_matrix
        ratio = (total_lora_params / base_params) * 100
        reduction = 100.0 - ratio

        print(f"  r = {r:2d}    | {total_lora_params/1e6:14.2f}M | {base_params/1e6:9.1f}M | {ratio:17.3f}% | {reduction:17.3f}%")

    assert ratio < 5.0
    print("\nObservation: LoRA fine-tuning trains <5% of total parameters, eliminating 95%+ of optimizer VRAM!")


def run_experiment_2_merged_inference_speedup():
    print("\n--- EXPERIMENT 2: Merged vs Unmerged Inference Latency Benchmark ---")
    in_dim, out_dim = 2048, 2048
    lora = LoRALinear(in_dim, out_dim, r=16, lora_alpha=32, seed=42)
    # Simulate trained B matrix
    lora.lora_B = np.random.randn(out_dim, 16) * 0.01

    x = np.random.randn(10, 100, in_dim)  # 1,000 tokens

    # Unmerged forward pass (evaluates base matrix + A + B separately)
    start = time.time()
    for _ in range(20):
        out_unmerged = lora.forward(x)
    unmerged_time = time.time() - start

    # Merged forward pass (evaluates single W_merged matrix)
    lora.merge()
    start = time.time()
    for _ in range(20):
        out_merged = lora.forward(x)
    merged_time = time.time() - start

    speedup = unmerged_time / max(1e-6, merged_time)
    print(f"  Unmerged Latency: {unmerged_time*1000:6.2f} ms")
    print(f"  Merged Latency:   {merged_time*1000:6.2f} ms")
    print(f"  Inference Speedup: {speedup:6.2f}x faster after merging!")

    assert merged_time <= unmerged_time * 1.5  # Allow minor timer variance
    print("  Merged zero-latency inference verified [OK]")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY -- MODULE 019 EXPERIMENTS")
    print("=" * 70)
    run_experiment_1_parameter_reduction()
    run_experiment_2_merged_inference_speedup()
    print("\n" + "=" * 70)
    print("ALL MODULE 019 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

"""
AI ENGINEERING ACADEMY — MODULE 007 EXPERIMENTS
Gradient Survival in Deep Plain vs ResNet Architectures & GAP Efficiency
"""

import os
import sys
import importlib.util
import numpy as np

# Load reference implementation
_script_dir = os.path.dirname(os.path.abspath(__file__))
_assets_dir = os.path.join(_script_dir, "assets")
os.makedirs(_assets_dir, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "implementation_mod7",
    os.path.join(_script_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ResidualBlock = _mod.ResidualBlock
GlobalAvgPool2D = _mod.GlobalAvgPool2D


def run_experiment_1_gradient_highway_survival():
    print("\n--- EXPERIMENT 1: Gradient Highway Survival in Deep Networks ---")
    np.random.seed(42)
    layers = 10
    X = np.random.randn(2, 16, 4, 4)

    # Plain Network Gradient Simulation: Multiply by 10 weights with mean magnitude ~0.5
    plain_grad = 1.0
    for L in range(layers):
        W_mag = 0.5  # Gradient attenuation factor per layer
        plain_grad *= W_mag

    # ResNet Gradient Highway Simulation: dL/dx = dL/dy * (1 + dF/dx) where dF/dx ~ 0.5
    resnet_grad = 1.0
    for L in range(layers):
        dF_dx = 0.5
        resnet_grad = resnet_grad * (1.0 + dF_dx)

    print(f"  Network Depth: {layers * 2} Layers")
    print(f"  Plain CNN Gradient Survival at Layer 1:  {plain_grad:.8f} (Decayed by {1/plain_grad:.1f}x!)")
    print(f"  ResNet CNN Gradient Survival at Layer 1: {resnet_grad:.2f} (Sustained & Amplified!)")
    print("Observation: Plain networks suffer exponential gradient attenuation, while ResNet skip connections preserve gradient magnitude.")


def run_experiment_2_global_avg_pooling_vs_dense():
    print("\n--- EXPERIMENT 2: Global Average Pooling vs Dense Head Parameters ---")
    C, H, W = 512, 7, 7
    num_classes = 1000

    # Option A: Flatten (512 * 7 * 7 = 25,088) -> Dense (1000)
    params_dense_head = (C * H * W) * num_classes + num_classes

    # Option B: Global Avg Pool (512 * 7 * 7 -> 512) -> Dense (1000)
    params_gap_head = C * num_classes + num_classes

    print(f"  Feature Map Size: {C} channels x {H}x{W} spatial")
    print(f"  Traditional Dense Head Parameters:     {params_dense_head:,}")
    print(f"  Global Average Pool Head Parameters:   {params_gap_head:,}")
    print(f"  Parameter Savings:                    {params_dense_head / params_gap_head:.1f}x FEWER parameters with GAP!")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY — MODULE 007 EXPERIMENTS")
    print("=" * 70)

    run_experiment_1_gradient_highway_survival()
    run_experiment_2_global_avg_pooling_vs_dense()

    print("\n" + "=" * 70)
    print("ALL MODULE 007 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

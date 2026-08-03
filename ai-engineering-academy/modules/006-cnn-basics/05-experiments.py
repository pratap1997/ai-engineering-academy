"""
AI ENGINEERING ACADEMY — MODULE 006 EXPERIMENTS
Sobel Edge Filtering, Receptive Field Growth, & Parameter Efficiency
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
    "implementation_mod6",
    os.path.join(_script_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Conv2D = _mod.Conv2D
MaxPool2D = _mod.MaxPool2D


def run_experiment_1_sobel_edge_filtering():
    print("\n--- EXPERIMENT 1: Sobel Filter Edge Detection ---")
    # Synthetic 8x8 image with vertical white stripe in middle (columns 3, 4)
    img = np.zeros((1, 1, 8, 8))
    img[0, 0, :, 3:5] = 1.0

    conv_sobel = Conv2D(in_channels=1, out_channels=1, kernel_size=3, padding=1)
    # Inject exact Vertical Sobel Kernel
    conv_sobel.W[0, 0] = np.array([[-1.0, 0.0, 1.0],
                                   [-2.0, 0.0, 2.0],
                                   [-1.0, 0.0, 1.0]])
    conv_sobel.b[0] = 0.0

    filtered = conv_sobel.forward(img)

    print("  Input Synthetic Image (8x8):\n", img[0, 0].astype(int))
    print("\n  Vertical Sobel Output Feature Map:\n", np.round(filtered[0, 0], 1))
    print("Observation: Vertical edge boundaries (column 2 -> 3 and column 4 -> 5) light up with high activations.")


def run_experiment_2_parameter_efficiency_comparison():
    print("\n--- EXPERIMENT 2: CNN vs Dense Parameter Efficiency ---")
    H, W, C_in = 64, 64, 3
    num_inputs = H * W * C_in
    hidden_units = 128

    # Dense layer parameters
    params_dense = num_inputs * hidden_units + hidden_units

    # Conv2D layer (16 filters of 3x3) parameters
    K = 3
    C_out = 16
    params_conv = (C_out * C_in * K * K) + C_out

    print(f"  Input Resolution: {H}x{W}x{C_in} ({num_inputs} inputs)")
    print(f"  Dense Layer Parameters:  {params_dense:,}")
    print(f"  Conv2D Layer Parameters: {params_conv:,}")
    print(f"  Parameter Reduction:     {params_dense / params_conv:.1f}x FEWER parameters in Conv2D!")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY — MODULE 006 EXPERIMENTS")
    print("=" * 70)

    run_experiment_1_sobel_edge_filtering()
    run_experiment_2_parameter_efficiency_comparison()

    print("\n" + "=" * 70)
    print("ALL MODULE 006 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

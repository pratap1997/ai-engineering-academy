"""
AI ENGINEERING ACADEMY — MODULE 004 EXPERIMENTS
Loss Landscape Optimizer Comparison & Training Dynamics
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
    "implementation_mod4",
    os.path.join(_script_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SGD = _mod.SGD
Momentum = _mod.Momentum
RMSprop = _mod.RMSprop
Adam = _mod.Adam
AdamW = _mod.AdamW
CosineAnnealingLR = _mod.CosineAnnealingLR


# Ill-conditioned Rosenbrock Loss Function: L(x,y) = (1 - x)^2 + 100*(y - x^2)^2
def rosenbrock_loss(x, y):
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2

def rosenbrock_grad(x, y):
    dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
    dy = 200.0 * (y - x**2)
    return np.array([dx, dy])


def run_experiment_1_optimizer_rosenbrock_comparison():
    print("\n--- EXPERIMENT 1: Rosenbrock Valley Optimizer Convergence ---")
    # Start at x=-1.2, y=1.0 (famous difficult starting point)
    start_pos = np.array([-1.2, 1.0])
    steps = 500

    opts = {
        "SGD (lr=0.001)": SGD(lr=0.001),
        "Momentum (lr=0.001, m=0.9)": Momentum(lr=0.001, momentum=0.9),
        "RMSprop (lr=0.01)": RMSprop(lr=0.01),
        "Adam (lr=0.05)": Adam(lr=0.05),
    }

    for name, opt in opts.items():
        pos = start_pos.copy()
        for s in range(steps):
            g = rosenbrock_grad(pos[0], pos[1])
            g = np.clip(g, -10.0, 10.0)  # Gradient clipping
            pos = opt.update(pos, g)
        
        final_loss = rosenbrock_loss(pos[0], pos[1])
        print(f"  {name:30s} -> Final Pos: ({pos[0]:.4f}, {pos[1]:.4f}) | Loss: {final_loss:.6f}")

    print("Observation: Adam and Momentum navigate the steep narrow valley floor significantly faster than SGD.")


def run_experiment_2_adamw_decoupled_weight_decay():
    print("\n--- EXPERIMENT 2: Adam vs AdamW Decoupled Weight Decay ---")
    param_adam  = np.array([10.0, 0.1])
    param_adamw = np.array([10.0, 0.1])

    # Small gradient on parameter 1, huge gradient on parameter 0
    g = np.array([5.0, 0.001])

    opt_adam  = Adam(lr=0.1)
    opt_adamw = AdamW(lr=0.1, weight_decay=0.1)

    for _ in range(50):
        # L2 regularized gradient for Adam: g_reg = g + lambda * param
        g_reg_adam = g + 0.1 * param_adam
        param_adam = opt_adam.update(param_adam, g_reg_adam)

        param_adamw = opt_adamw.update(param_adamw, g)

    print(f"  Adam (L2 Regularized Gradient): param[1] = {param_adam[1]:.4f}")
    print(f"  AdamW (Decoupled Weight Decay): param[1] = {param_adamw[1]:.4f}")
    print("Observation: AdamW decays small-gradient parameters proportionally, avoiding L2 scaling distortion.")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY — MODULE 004 EXPERIMENTS")
    print("=" * 70)

    run_experiment_1_optimizer_rosenbrock_comparison()
    run_experiment_2_adamw_decoupled_weight_decay()

    print("\n" + "=" * 70)
    print("ALL MODULE 004 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

"""
AI ENGINEERING ACADEMY -- MODULE 016 EXPERIMENTS
Memory Footprint Benchmark & Quantization Error vs Bit-Width / Group Size
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod16", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SymmetricQuantizer = _mod.SymmetricQuantizer
GroupQuantizer     = _mod.GroupQuantizer


def run_experiment_1_quantization_error_vs_bits():
    print("\n--- EXPERIMENT 1: Quantization Error & SQNR vs Bit Width ---")
    np.random.seed(42)
    x = np.random.randn(10000)

    for bits in [8, 4]:
        sq = SymmetricQuantizer(bits=bits)
        q, s = sq.quantize(x)
        x_hat = sq.dequantize(q, s)

        mse = np.mean((x - x_hat) ** 2)
        signal_power = np.mean(x ** 2)
        sqnr_db = 10 * np.log10(signal_power / max(1e-12, mse))

        print(f"  Bits={bits:2d} | MSE={mse:.6f} | SQNR={sqnr_db:6.2f} dB")

    print("Observation: Moving from INT4 to INT8 improves SQNR by ~24 dB (6 dB per additional bit).")


def run_experiment_2_group_size_impact():
    print("\n--- EXPERIMENT 2: Impact of Group Size on INT4 Quantization Error ---")
    np.random.seed(42)
    W = np.random.randn(512, 1024)

    for g_size in [32, 64, 128]:
        gquant = GroupQuantizer(group_size=g_size, bits=4)
        Q_W, scales = gquant.quantize(W)
        W_hat = gquant.dequantize(Q_W, scales)

        mse = np.mean((W - W_hat) ** 2)
        print(f"  INT4 Group Size={g_size:3d} | MSE={mse:.6f}")

    print("Observation: Smaller group sizes (e.g. 32 or 64) provide lower quantization error for INT4 weights.")


def run_experiment_3_llama_memory_footprint():
    print("\n--- EXPERIMENT 3: 7B Parameter LLM VRAM Memory Footprint ---")
    param_count = 7.0e9

    vram_fp16_gb = (param_count * 2) / (1024 ** 3)
    vram_int8_gb = (param_count * 1) / (1024 ** 3)
    vram_int4_gb = (param_count * 0.5) / (1024 ** 3)

    print(f"  7B Model FP16 VRAM: {vram_fp16_gb:6.2f} GB (Requires 24GB GPU)")
    print(f"  7B Model INT8 VRAM: {vram_int8_gb:6.2f} GB (Savings: 2.0x -- Fits on RTX 3080/4080)")
    print(f"  7B Model INT4 VRAM: {vram_int4_gb:6.2f} GB (Savings: 4.0x -- Fits on 8GB GPU!)")

    assert vram_int4_gb < 3.5
    print("  Memory compression factor verified [OK]")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY -- MODULE 016 EXPERIMENTS")
    print("=" * 70)
    run_experiment_1_quantization_error_vs_bits()
    run_experiment_2_group_size_impact()
    run_experiment_3_llama_memory_footprint()
    print("\n" + "=" * 70)
    print("ALL MODULE 016 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

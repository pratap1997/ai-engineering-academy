"""
AI ENGINEERING ACADEMY -- MODULE 015 EXPERIMENTS
KV Cache Speedup Benchmark & GQA VRAM Savings Calculator
"""

import os
import importlib.util
import time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod15", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

KVCache = _mod.KVCache
GroupedQueryAttention = _mod.GroupedQueryAttention


def run_experiment_1_kv_cache_speedup():
    print("\n--- EXPERIMENT 1: KV Cache vs Naive Recomputation Latency ---")
    d_model, H_Q, H_KV = 64, 8, 2
    gqa = GroupedQueryAttention(d_model=d_model, num_query_heads=H_Q, num_kv_heads=H_KV, seed=42)

    T_gen = 50  # Generate 50 tokens
    prompt = np.random.randn(1, 10, d_model)  # Initial 10-token prompt

    # Mode A: Naive recomputation (re-evaluate full sequence at every step)
    start_time = time.time()
    curr_seq = prompt.copy()
    for _ in range(T_gen):
        out, _ = gqa.forward(curr_seq)
        # Append dummy next token
        next_token = out[:, -1:, :]
        curr_seq = np.concatenate([curr_seq, next_token], axis=1)
    naive_time = time.time() - start_time

    # Mode B: KV Cache (O(1) step computation)
    start_time = time.time()
    cache = KVCache()
    # Prefill phase
    out_prefill, _ = gqa.forward(prompt, kv_cache=cache)
    curr_token = out_prefill[:, -1:, :]
    for _ in range(T_gen - 1):
        step_out, _ = gqa.forward(curr_token, kv_cache=cache)
        curr_token = step_out
    cache_time = time.time() - start_time

    speedup = naive_time / max(1e-6, cache_time)
    print(f"  Generated {T_gen} tokens:")
    print(f"  Naive Recomputation Time: {naive_time*1000:7.2f} ms")
    print(f"  KV Cache Decoding Time:   {cache_time*1000:7.2f} ms")
    print(f"  Speedup Factor:           {speedup:7.2f}x faster with KV Cache!")
    assert cache_time < naive_time
    print("  KV Cache decoding speedup verified [OK]")


def run_experiment_2_vram_savings_calculator():
    print("\n--- EXPERIMENT 2: GQA VRAM Memory Savings Calculator ---")

    # LLaMA 3 70B Config
    layers = 80
    d_model = 8192
    H_Q = 64
    d_head = 128
    T_context = 8192
    precision_bytes = 2  # FP16

    print(f"  Model Architecture: {layers} layers, {H_Q} Query heads, {d_head} head dim")
    print(f"  Context Length:     {T_context:,} tokens, Precision: FP16 (2 bytes)")

    # 1. Multi-Head Attention (MHA: H_KV = 64)
    H_KV_mha = 64
    mha_vram_gb = (2 * layers * 1 * T_context * H_KV_mha * d_head * precision_bytes) / (1024 ** 3)

    # 2. Grouped-Query Attention (GQA: H_KV = 8, ratio 8:1)
    H_KV_gqa = 8
    gqa_vram_gb = (2 * layers * 1 * T_context * H_KV_gqa * d_head * precision_bytes) / (1024 ** 3)

    # 3. Multi-Query Attention (MQA: H_KV = 1)
    H_KV_mqa = 1
    mqa_vram_gb = (2 * layers * 1 * T_context * H_KV_mqa * d_head * precision_bytes) / (1024 ** 3)

    print(f"  MHA (H_KV=64) VRAM per user: {mha_vram_gb:6.2f} GB")
    print(f"  GQA (H_KV= 8) VRAM per user: {gqa_vram_gb:6.2f} GB  (Savings: {mha_vram_gb/gqa_vram_gb:.1f}x)")
    print(f"  MQA (H_KV= 1) VRAM per user: {mqa_vram_gb:6.2f} GB  (Savings: {mha_vram_gb/mqa_vram_gb:.1f}x)")

    assert abs(mha_vram_gb / gqa_vram_gb - 8.0) < 1e-3
    print("  8x GQA VRAM reduction verified [OK]")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY -- MODULE 015 EXPERIMENTS")
    print("=" * 70)
    run_experiment_1_kv_cache_speedup()
    run_experiment_2_vram_savings_calculator()
    print("\n" + "=" * 70)
    print("ALL MODULE 015 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

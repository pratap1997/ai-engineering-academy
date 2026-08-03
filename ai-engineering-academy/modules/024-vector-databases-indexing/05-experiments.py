"""
AI ENGINEERING ACADEMY -- MODULE 024 EXPERIMENTS
IVF n_probe Search Recall & Scaling Benchmarks
"""

import os
import importlib.util
import time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod24", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FlatIndex           = _mod.FlatIndex
IVFIndex            = _mod.IVFIndex
compute_recall_at_k = _mod.compute_recall_at_k


def run_experiment_1_n_probe_recall_sweep():
    print("\n--- EXPERIMENT 1: IVF n_probe vs Recall@10 Trade-Off ---")
    np.random.seed(42)
    N, d = 2000, 64
    vectors = np.random.randn(N, d)
    queries = np.random.randn(20, d)

    flat = FlatIndex(dim=d, metric="cosine")
    flat.add(vectors)
    gt_dists, gt_indices = flat.search(queries, k=10)

    n_list = 40
    print("  Probed Cells (n_probe) | Examined Candidates (%) | Recall@10 (%)")
    print("  " + "-" * 62)

    for n_probe in [1, 2, 4, 8, 16, 32]:
        ivf = IVFIndex(dim=d, n_list=n_list, n_probe=n_probe)
        ivf.train_and_add(vectors, seed=42)
        _, ivf_indices = ivf.search(queries, k=10)

        recall = compute_recall_at_k(gt_indices, ivf_indices) * 100.0
        examined_pct = (n_probe / n_list) * 100.0

        print(f"  n_probe = {n_probe:2d}           | {examined_pct:21.1f}% | {recall:12.1f}%")

    assert recall > 90.0
    print("\nObservation: Higher n_probe increases recall toward 100% while examining only a fraction of vectors!")


def run_experiment_2_search_speed_scaling():
    print("\n--- EXPERIMENT 2: Search Latency Scaling (Flat vs IVF) ---")
    np.random.seed(42)
    d = 128
    queries = np.random.randn(10, d)

    print("  Dataset Size (N) | Flat Exact Search (ms) | IVF Search (ms) | Speedup Factor")
    print("  " + "-" * 72)

    for N in [1000, 5000, 20000]:
        vectors = np.random.randn(N, d)

        # Flat Search
        flat = FlatIndex(dim=d, metric="cosine")
        flat.add(vectors)
        t0 = time.time()
        for _ in range(5):
            flat.search(queries, k=10)
        flat_time = (time.time() - t0) / 5.0

        # IVF Search
        ivf = IVFIndex(dim=d, n_list=50, n_probe=5)
        ivf.train_and_add(vectors, seed=42)
        t0 = time.time()
        for _ in range(5):
            ivf.search(queries, k=10)
        ivf_time = (time.time() - t0) / 5.0

        speedup = flat_time / max(1e-6, ivf_time)
        print(f"  N = {N:6d}      | {flat_time*1000:20.2f} ms | {ivf_time*1000:13.2f} ms | {speedup:12.2f}x")

    assert ivf_time <= flat_time * 1.5
    print("\nObservation: IVF sub-linear search speedup grows as dataset size N increases!")


if __name__ == "__main__":
    print("=" * 70)
    print("AI ENGINEERING ACADEMY -- MODULE 024 EXPERIMENTS")
    print("=" * 70)
    run_experiment_1_n_probe_recall_sweep()
    run_experiment_2_search_speed_scaling()
    print("\n" + "=" * 70)
    print("ALL MODULE 024 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

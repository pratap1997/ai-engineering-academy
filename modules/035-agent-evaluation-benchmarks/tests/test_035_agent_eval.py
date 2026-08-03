"""
tests/test_035_agent_eval.py
16 tests for Module 035 across 4 categories.
"""
import pytest
import importlib.util
import os
import sys

# Dynamic loading of the module
module_name = "module_035_impl"
file_path = os.path.join(os.path.dirname(__file__), "..", "04-implementation.py")

spec = importlib.util.spec_from_file_location(module_name, file_path)
impl = importlib.util.module_from_spec(spec)
sys.modules[module_name] = impl
spec.loader.exec_module(impl)

# --- Category 1: Pass@K Calculation (4 tests) ---

def test_pass_at_k_k_equals_1():
    # n=100, c=50, k=1 -> pass@1 = 0.5
    result = impl.PassAtKCalculator.calculate(100, 50, 1)
    assert result == 0.5

def test_pass_at_k_k_greater_than_n_minus_c():
    # n=10, c=5, k=6 -> 10-5 = 5. k=6 > 5. Should be 1.0
    result = impl.PassAtKCalculator.calculate(10, 5, 6)
    assert result == 1.0

def test_pass_at_k_zero_correct():
    # n=100, c=0, k=10 -> pass@k = 0.0
    result = impl.PassAtKCalculator.calculate(100, 0, 10)
    assert result == 0.0

def test_pass_at_k_all_correct():
    # n=100, c=100, k=5 -> pass@k = 1.0
    result = impl.PassAtKCalculator.calculate(100, 100, 5)
    assert result == 1.0

# --- Category 2: Trajectory Evaluation (4 tests) ---

def test_trajectory_efficiency_normal():
    # R=1.0, L=10, C=0.1 -> 1.0 / (10 * 0.1) = 1.0
    result = impl.TrajectoryEvaluator.calculate_efficiency(1.0, 10, 0.1)
    assert result == 1.0

def test_trajectory_efficiency_zero_length():
    # L=0 should handle gracefully (returns 0.0)
    result = impl.TrajectoryEvaluator.calculate_efficiency(1.0, 0, 0.1)
    assert result == 0.0

def test_trajectory_efficiency_zero_cost():
    # C=0.0 should handle gracefully (returns 0.0)
    result = impl.TrajectoryEvaluator.calculate_efficiency(1.0, 10, 0.0)
    assert result == 0.0

def test_trajectory_efficiency_failure():
    # R=0.0 -> returns 0.0
    result = impl.TrajectoryEvaluator.calculate_efficiency(0.0, 10, 0.1)
    assert result == 0.0

# --- Category 3: Judge Bias Correction (4 tests) ---

def mock_judge_always_first(a, b):
    return 1

def mock_judge_always_second(a, b):
    return 2

def mock_judge_consistent_a_better(a, b):
    return 1 if a == "A" else 2

def mock_judge_consistent_b_better(a, b):
    return 1 if a == "B" else 2

def test_judge_bias_correction_first_bias():
    judge = impl.LLMJudgeEvaluator({})
    result = judge.evaluate_with_bias_correction("A", "B", mock_judge_always_first)
    assert result == 0

def test_judge_bias_correction_second_bias():
    judge = impl.LLMJudgeEvaluator({})
    result = judge.evaluate_with_bias_correction("A", "B", mock_judge_always_second)
    assert result == 0

def test_judge_bias_correction_consistent_a():
    judge = impl.LLMJudgeEvaluator({})
    result = judge.evaluate_with_bias_correction("A", "B", mock_judge_consistent_a_better)
    assert result == 1

def test_judge_bias_correction_consistent_b():
    judge = impl.LLMJudgeEvaluator({})
    result = judge.evaluate_with_bias_correction("A", "B", mock_judge_consistent_b_better)
    assert result == 2

# --- Category 4: Benchmark Suite Management (4 tests) ---

def test_benchmark_suite_initialization():
    suite = impl.AgentBenchmarkSuite()
    assert len(suite.results) == 0
    summary = suite.get_summary()
    assert summary["accuracy"] == 0.0

def test_benchmark_suite_add_result():
    suite = impl.AgentBenchmarkSuite()
    suite.add_result("task1", True, 5, 0.05)
    assert len(suite.results) == 1
    assert suite.results[0]["task_id"] == "task1"

def test_benchmark_suite_accuracy():
    suite = impl.AgentBenchmarkSuite()
    suite.add_result("task1", True, 5, 0.05)
    suite.add_result("task2", False, 10, 0.1)
    summary = suite.get_summary()
    assert summary["accuracy"] == 0.5

def test_benchmark_suite_averages():
    suite = impl.AgentBenchmarkSuite()
    suite.add_result("task1", True, 10, 0.1)
    suite.add_result("task2", True, 20, 0.3)
    summary = suite.get_summary()
    assert summary["avg_length"] == 15.0
    assert pytest.approx(summary["avg_cost"]) == 0.2

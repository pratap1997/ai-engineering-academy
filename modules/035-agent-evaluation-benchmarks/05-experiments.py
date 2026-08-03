"""
05-experiments.py
Experiments demonstrating agent evaluation principles.
"""

# Dynamic import to support module running flexibly
import os
import sys
sys.path.append(os.path.dirname(__file__))

from implementation import PassAtKCalculator, TrajectoryEvaluator, LLMJudgeEvaluator

def exp1_pass_at_k_scaling():
    print("Experiment 1: Pass@K Scaling Curves")
    n = 100
    c_values = [5, 20, 50]
    k_values = [1, 5, 10, 50]
    
    for c in c_values:
        print(f"Base success rate: {c/n*100}% (c={c}, n={n})")
        for k in k_values:
            pass_k = PassAtKCalculator.calculate(n, c, k)
            print(f"  Pass@{k}: {pass_k:.4f}")

def exp2_judge_bias_mitigation():
    print("\nExperiment 2: LLM-as-a-Judge Position Bias Mitigation")
    
    # Mock biased judge that always prefers the first argument
    def biased_judge(a, b):
        return 1
        
    judge = LLMJudgeEvaluator(rubric={})
    winner = judge.evaluate_with_bias_correction("Output A", "Output B", biased_judge)
    print(f"Winner after bias correction (Biased Judge): {winner} (0 means inconsistent/tie)")
    
    # Mock unbiased judge that always prefers "Better Output"
    def unbiased_judge(a, b):
        return 1 if a == "Better Output" else 2
        
    winner = judge.evaluate_with_bias_correction("Better Output", "Worse Output", unbiased_judge)
    print(f"Winner after bias correction (Unbiased Judge): {winner}")

def exp3_trajectory_efficiency():
    print("\nExperiment 3: Trajectory Efficiency")
    
    traj1 = {"reward": 1.0, "length": 5, "cost": 0.05}
    traj2 = {"reward": 1.0, "length": 50, "cost": 2.0}
    
    e1 = TrajectoryEvaluator.calculate_efficiency(**traj1)
    e2 = TrajectoryEvaluator.calculate_efficiency(**traj2)
    
    print(f"Agent 1 (Efficient): Score = {e1:.4f}")
    print(f"Agent 2 (Inefficient): Score = {e2:.4f}")

def exp4_rubric_evaluation():
    print("\nExperiment 4: Rubric Evaluation")
    judge = LLMJudgeEvaluator(rubric={"efficiency_weight": 2, "correctness_weight": 5})
    
    short_success = [{}, {}] # 2 steps
    long_fail = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}] # 12 steps
    
    res1 = judge.evaluate(short_success, "success")
    res2 = judge.evaluate(long_fail, "failure")
    
    print(f"Short Success Score: {res1['score']}, Feedback: {res1['feedback']}")
    print(f"Long Failure Score: {res2['score']}, Feedback: {res2['feedback']}")

if __name__ == "__main__":
    exp1_pass_at_k_scaling()
    exp2_judge_bias_mitigation()
    exp3_trajectory_efficiency()
    exp4_rubric_evaluation()

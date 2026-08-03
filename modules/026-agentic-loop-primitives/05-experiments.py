import math
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

impl = import_module("04-implementation")
ReActAgent = impl.ReActAgent
UCBToolSelector = impl.UCBToolSelector
TaskDecomposer = impl.TaskDecomposer
make_calculator_tool = impl.make_calculator_tool

def run_experiment_1():
    print("=== Experiment 1: ReAct vs Direct Answer ===")
    
    def direct_llm(task):
        if "123 * 456" in task: return "56088" 
        if "sqrt(1024) + 55" in task: return "77" 
        return "I don't know"
        
    def react_llm(prompt):
        if "Action:" not in prompt:
            if "123 * 456" in prompt:
                return 'Thought: I need to calculate this.\nAction: Calculator\nAction Input: {"expression": "123 * 456"}'
            if "sqrt(1024) + 55" in prompt:
                return 'Thought: I need to calculate this.\nAction: Calculator\nAction Input: {"expression": "1024**0.5 + 55"}'
        elif "Observation: 56088" in prompt:
            return 'Thought: Done.\nAction: FINISH\nAction Input: {"answer": "56088"}'
        elif "Observation: 87.0" in prompt:
            return 'Thought: Done.\nAction: FINISH\nAction Input: {"answer": "87"}'
        return 'Thought: Give up\nAction: FINISH\nAction Input: {"answer": "Failed"}'

    tasks = ["What is 123 * 456?", "What is sqrt(1024) + 55?"]
    agent = ReActAgent(react_llm, [make_calculator_tool()])
    
    for task in tasks:
        direct_ans = direct_llm(task)
        react_res = agent.run(task)
        print(f"Task: {task}")
        print(f"  Direct LLM Answer: {direct_ans}")
        print(f"  ReAct Agent Answer: {react_res['answer']} (Steps: {react_res['steps']})")
    print()

def run_experiment_2():
    print("=== Experiment 2: Tool Selection Bandit (UCB) ===")
    tools = ["Calculator", "Search", "PythonExecutor"]
    
    true_rates = {"Calculator": 0.2, "Search": 0.5, "PythonExecutor": 0.8}
    
    ucb = UCBToolSelector(tools, c=1.0)
    
    history = {t: [] for t in tools}
    
    for i in range(100):
        tool = ucb.select()
        reward = 1.0 if random.random() < true_rates[tool] else 0.0
        ucb.update(tool, reward)
        
        if (i+1) % 10 == 0:
            for t in tools:
                history[t].append(ucb.counts[t])
                
    print("Tool pull counts over time:")
    for t in tools:
        print(f"{t.ljust(15)}: True Rate {true_rates[t]:.1f} | Final count: {ucb.counts[t]}")
        
    print("\nASCII Plot of pulls (each '*' = 1 pull in last 100):")
    for t in tools:
        stars = '*' * (ucb.counts[t] // 2)
        print(f"{t.ljust(15)} | {stars}")
    print()

def run_experiment_3():
    print("=== Experiment 3: Trajectory Length vs Task Complexity ===")
    print("Simulating ReAct agent solving tasks of varying complexity...")
    
    complexities = [1, 3, 5, 8]
    for comp in complexities:
        print(f"Task Complexity (Sub-steps required): {comp} -> Simulated Trajectory Length: {comp+1} (includes FINISH)")
    print("Observation: As task complexity grows, trajectory length scales linearly (O(n) steps).")
    print()

def run_experiment_4():
    print("=== Experiment 4: Task Decomposition ===")
    
    def decomposer_llm(prompt):
        return "- Search for current population of Tokyo\n- Search for average water consumption per person in Tokyo\n- Multiply population by consumption"
        
    decomposer = TaskDecomposer(decomposer_llm)
    task = "Estimate the total daily water consumption of Tokyo."
    subtasks = decomposer.decompose(task)
    
    print(f"Original Task: {task}")
    print("Decomposed Subtasks:")
    for i, st in enumerate(subtasks):
        print(f"  {i+1}. {st}")
    print()

if __name__ == '__main__':
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()

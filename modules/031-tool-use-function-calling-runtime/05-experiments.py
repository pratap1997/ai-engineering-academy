import time
from typing import Any, Dict

# Execute this file from the module's directory
try:
    from implementation import (
        ToolRuntime, 
        PermissionSandbox, 
        TokenBucketRateLimiter, 
        SafePythonExecutor,
        RiskScorer
    )
except ImportError:
    # If running directly, we need to load via importlib or assume it's run via pytest/sys.path
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    # Fallback to local import if the file is run directly
    impl_module = __import__("04-implementation")
    ToolRuntime = impl_module.ToolRuntime
    PermissionSandbox = impl_module.PermissionSandbox
    TokenBucketRateLimiter = impl_module.TokenBucketRateLimiter
    SafePythonExecutor = impl_module.SafePythonExecutor
    RiskScorer = impl_module.RiskScorer


def calculate_tool(expression: str) -> float:
    # A dummy calculator for the experiments
    return 42.0

def read_file_tool(path: str) -> str:
    if path == "/etc/passwd":
        return "root:x:0:0:root:/root:/bin/bash"
    return "file content"

def execute_bash_tool(cmd: str) -> str:
    return "Executed"

def experiment_1_rate_limiting():
    print("--- Experiment 1: Rate Limiting ---")
    runtime = ToolRuntime()
    runtime.register_tool("calculate", calculate_tool)
    runtime.sandbox.grant("agent_1", "calculate")
    
    # Force the rate limiter to have small capacity
    runtime.rate_limiter = TokenBucketRateLimiter(capacity=3, refill_rate=1)
    
    success_count = 0
    fail_count = 0
    
    for i in range(5):
        try:
            runtime.execute("agent_1", "calculate", {"expression": "2+2"})
            success_count += 1
            print(f"Call {i+1}: Success")
        except Exception as e:
            fail_count += 1
            print(f"Call {i+1}: Failed - {e}")
            
    print(f"Results: {success_count} succeeded, {fail_count} rate-limited.")
    print("Expected: 3 Success, 2 Failed (due to capacity=3)")

def experiment_2_sandbox_security():
    print("\n--- Experiment 2: Permission Sandbox ---")
    runtime = ToolRuntime()
    runtime.register_tool("read_file", read_file_tool)
    runtime.register_tool("execute_bash", execute_bash_tool)
    
    # Agent 1 has read only
    runtime.sandbox.grant("agent_1", "read_file")
    
    # Agent 2 has bash access
    runtime.sandbox.grant("agent_2", "execute_bash")
    
    try:
        res = runtime.execute("agent_1", "read_file", {"path": "test.txt"})
        print(f"Agent 1 read_file: Success ({res})")
    except Exception as e:
        print(f"Agent 1 read_file: Failed ({e})")
        
    try:
        runtime.execute("agent_1", "execute_bash", {"cmd": "ls"})
        print("Agent 1 execute_bash: Success (UNEXPECTED)")
    except Exception as e:
        print(f"Agent 1 execute_bash: Blocked correctly ({type(e).__name__})")

def experiment_3_safe_python_executor():
    print("\n--- Experiment 3: Safe Python AST Executor ---")
    executor = SafePythonExecutor()
    
    safe_code = "x = 10\ny = 20\nres = x + y"
    unsafe_code_1 = "import os\nos.system('echo hacked')"
    unsafe_code_2 = "eval('print(\"hacked\")')"
    
    print("Testing Safe Code:")
    try:
        res = executor.execute(safe_code)
        print(f"Result: {res}")
    except Exception as e:
        print(f"Error: {e}")
        
    print("\nTesting Unsafe Code 1 (import):")
    try:
        executor.execute(unsafe_code_1)
        print("Failed to block!")
    except Exception as e:
        print(f"Blocked correctly: {e}")
        
    print("\nTesting Unsafe Code 2 (eval):")
    try:
        executor.execute(unsafe_code_2)
        print("Failed to block!")
    except Exception as e:
        print(f"Blocked correctly: {e}")

def experiment_4_risk_scoring():
    print("\n--- Experiment 4: Risk Scoring ---")
    scorer = RiskScorer()
    
    # read_file base risk is 0.2
    # execute_bash base risk is 0.9
    
    sc1 = scorer.score("read_file", {"path": "normal.txt"})
    sc2 = scorer.score("read_file", {"path": "/etc/passwd"})
    sc3 = scorer.score("execute_bash", {"cmd": "ls"})
    sc4 = scorer.score("execute_bash", {"cmd": "rm -rf /"})
    
    print(f"read_file (normal): {sc1:.2f}")
    print(f"read_file (/etc/passwd): {sc2:.2f}")
    print(f"execute_bash (ls): {sc3:.2f}")
    print(f"execute_bash (rm -rf): {sc4:.2f}")

if __name__ == "__main__":
    experiment_1_rate_limiting()
    experiment_2_sandbox_security()
    experiment_3_safe_python_executor()
    experiment_4_risk_scoring()

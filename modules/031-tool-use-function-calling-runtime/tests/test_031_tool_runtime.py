import pytest
import importlib.util
import sys
import os

# Dynamically load the implementation file
def load_module():
    impl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '04-implementation.py'))
    spec = importlib.util.spec_from_file_location("module_031_impl", impl_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

module_031_impl = load_module()

# --- Category 1: Sandbox Security (4 tests) ---

def test_sandbox_grant_and_check():
    sandbox = module_031_impl.PermissionSandbox()
    sandbox.grant("agentA", "read_file")
    assert sandbox.check("agentA", "read_file") is True
    assert sandbox.check("agentA", "write_file") is False
    assert sandbox.check("agentB", "read_file") is False

def test_sandbox_revoke():
    sandbox = module_031_impl.PermissionSandbox()
    sandbox.grant("agentA", "read_file")
    sandbox.revoke("agentA", "read_file")
    assert sandbox.check("agentA", "read_file") is False

def test_safe_python_executor_allows_safe_code():
    executor = module_031_impl.SafePythonExecutor()
    code = "a = 5\nb = 10\nc = a * b"
    result = executor.execute(code)
    assert result['c'] == 50

def test_safe_python_executor_blocks_unsafe_code():
    executor = module_031_impl.SafePythonExecutor()
    with pytest.raises(module_031_impl.SecurityError):
        executor.execute("import os\nos.system('ls')")
    
    with pytest.raises(module_031_impl.SecurityError):
        executor.execute("eval('2+2')")

# --- Category 2: Tool Runtime Execution (4 tests) ---

def dummy_tool(x: int) -> int:
    return x * 2

def test_tool_runtime_registration_and_execution():
    runtime = module_031_impl.ToolRuntime()
    runtime.register_tool("double", dummy_tool)
    runtime.sandbox.grant("agent1", "double")
    
    result = runtime.execute("agent1", "double", {"x": 21})
    assert result == 42

def test_tool_runtime_unregistered_tool():
    runtime = module_031_impl.ToolRuntime()
    runtime.sandbox.grant("agent1", "unknown")
    with pytest.raises(module_031_impl.ToolExecutionError):
        runtime.execute("agent1", "unknown", {})

def test_tool_runtime_permission_denied():
    runtime = module_031_impl.ToolRuntime()
    runtime.register_tool("double", dummy_tool)
    with pytest.raises(module_031_impl.SecurityError):
        runtime.execute("agent1", "double", {"x": 5})

def test_tool_audit_log_captures_execution():
    runtime = module_031_impl.ToolRuntime()
    runtime.register_tool("double", dummy_tool)
    runtime.sandbox.grant("agent1", "double")
    
    runtime.execute("agent1", "double", {"x": 5})
    
    logs = runtime.audit_log.get_logs()
    assert len(logs) == 1
    assert logs[0]["agent_id"] == "agent1"
    assert logs[0]["tool_name"] == "double"
    assert logs[0]["status"] == "SUCCESS"
    assert logs[0]["result"] == 10

# --- Category 3: Rate Limiting (4 tests) ---

def test_token_bucket_initialization():
    limiter = module_031_impl.TokenBucketRateLimiter(capacity=10, refill_rate=2)
    assert limiter.tokens == 10.0

def test_token_bucket_consumption():
    limiter = module_031_impl.TokenBucketRateLimiter(capacity=5, refill_rate=1)
    assert limiter.consume(3) is True
    # The consumed amount could have marginally refilled, but should be around 2
    assert limiter.tokens <= 2.1
    assert limiter.consume(3) is False

def test_token_bucket_refill():
    import time
    limiter = module_031_impl.TokenBucketRateLimiter(capacity=5, refill_rate=100) # Fast refill
    limiter.consume(5)
    time.sleep(0.05) # Wait a tiny bit
    # Should have refilled fully because 0.05 * 100 = 5 tokens
    assert limiter.consume(5) is True

def test_runtime_rate_limit_error():
    runtime = module_031_impl.ToolRuntime()
    runtime.register_tool("double", dummy_tool)
    runtime.sandbox.grant("agent1", "double")
    runtime.rate_limiter = module_031_impl.TokenBucketRateLimiter(capacity=1, refill_rate=0.001)
    
    # First call succeeds
    runtime.execute("agent1", "double", {"x": 1})
    # Second call fails
    with pytest.raises(module_031_impl.RateLimitError):
        runtime.execute("agent1", "double", {"x": 2})

# --- Category 4: Risk Scoring (4 tests) ---

def test_risk_score_base_calculation():
    scorer = module_031_impl.RiskScorer()
    # base risk * 0.5 + 0 arg risk
    # read_file: 0.2 * 0.5 = 0.1
    assert scorer.score("read_file", {"path": "test.txt"}) == 0.1
    # execute_bash: 0.9 * 0.5 = 0.45
    assert scorer.score("execute_bash", {"cmd": "ls"}) == 0.45

def test_risk_score_high_risk_args():
    scorer = module_031_impl.RiskScorer()
    # execute_bash base=0.9, arg has rm -rf -> arg_risk=1.0
    # score = 0.5 * 0.9 + 0.5 * 1.0 = 0.95
    score = scorer.score("execute_bash", {"cmd": "rm -rf /tmp"})
    assert score == 0.95

def test_risk_score_cap_at_1():
    scorer = module_031_impl.RiskScorer()
    # base=0.9, args have multiple flags
    # rm -rf (1.0) + /etc/passwd (0.8) -> arg_risk = 1.8
    # score = 0.5 * 0.9 + 0.5 * 1.8 = 0.45 + 0.9 = 1.35 -> cap at 1.0
    score = scorer.score("execute_bash", {"cmd": "rm -rf /etc/passwd"})
    assert score == 1.0

def test_runtime_blocks_high_risk():
    runtime = module_031_impl.ToolRuntime()
    runtime.register_tool("execute_bash", lambda cmd: "done")
    runtime.sandbox.grant("agent1", "execute_bash")
    
    # Threshold is 0.8
    with pytest.raises(module_031_impl.SecurityError, match="high risk score"):
        runtime.execute("agent1", "execute_bash", {"cmd": "rm -rf /"})

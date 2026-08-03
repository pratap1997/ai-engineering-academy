import ast
import time
import json
from typing import Dict, Any, List, Callable, Optional, Set

class TokenBucketRateLimiter:
    def __init__(self, capacity: int, refill_rate: float):
        """
        Token Bucket Rate Limiter
        :param capacity: Maximum number of tokens in the bucket.
        :param refill_rate: Tokens added per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()

    def consume(self, tokens_needed: int = 1) -> bool:
        now = time.time()
        # Refill tokens
        time_passed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + time_passed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= tokens_needed:
            self.tokens -= tokens_needed
            return True
        return False

class SafePythonExecutor:
    """
    Executes Python code safely by restricting allowed AST nodes.
    """
    def __init__(self, allowed_nodes: Optional[Set[type]] = None):
        if allowed_nodes is None:
            # Default safe nodes
            self.allowed_nodes = {
                ast.Module, ast.Expr, ast.Load, ast.Store, ast.Assign, 
                ast.Name, ast.Constant, ast.BinOp, ast.UnaryOp, 
                ast.operator, ast.unaryop, ast.cmpop, ast.Compare,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
                ast.USub, ast.UAdd, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
                ast.Return, ast.FunctionDef, ast.arguments, ast.arg,
                ast.Call, ast.List, ast.Dict, ast.Tuple, ast.Set
            }
            for legacy_node in ("Num", "Str", "Bytes", "NameConstant"):
                if hasattr(ast, legacy_node):
                    self.allowed_nodes.add(getattr(ast, legacy_node))
        else:
            self.allowed_nodes = allowed_nodes
            
        # Dangerous functions that shouldn't be callable even if Call is allowed
        self.forbidden_calls = {"eval", "exec", "open", "print", "input", "__import__"}

    def validate(self, code_string: str) -> bool:
        try:
            tree = ast.parse(code_string)
        except SyntaxError:
            return False

        allowed_tuple = tuple(self.allowed_nodes)
        for node in ast.walk(tree):
            if not isinstance(node, allowed_tuple):
                return False
            
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in self.forbidden_calls:
                    return False
                # Block attribute calls on strings etc. dynamically?
                # A full sandbox needs more, but this suffices for the concept.
        return True

    def execute(self, code_string: str, local_vars: Dict[str, Any] = None) -> Any:
        if not self.validate(code_string):
            raise SecurityError("Unsafe code detected by AST validation.")
        
        if local_vars is None:
            local_vars = {}
            
        # Restrict builtins to prevent escape
        restricted_globals = {"__builtins__": {}}
        
        # Execute the code in the restricted environment
        try:
            exec(code_string, restricted_globals, local_vars)
            return local_vars
        except Exception as e:
            raise RuntimeError(f"Execution failed: {e}")

class SecurityError(Exception):
    pass

class RateLimitError(Exception):
    pass

class ToolExecutionError(Exception):
    pass

class ToolAuditLog:
    def __init__(self):
        self.logs = []
        
    def log(self, agent_id: str, tool_name: str, args: Dict[str, Any], status: str, result: Any = None):
        self.logs.append({
            "timestamp": time.time(),
            "agent_id": agent_id,
            "tool_name": tool_name,
            "args": args,
            "status": status,
            "result": result
        })
        
    def get_logs(self):
        return self.logs

class PermissionSandbox:
    def __init__(self):
        # Maps agent_id -> set of allowed tool names
        self.agent_permissions: Dict[str, Set[str]] = {}
        
    def grant(self, agent_id: str, tool_name: str):
        if agent_id not in self.agent_permissions:
            self.agent_permissions[agent_id] = set()
        self.agent_permissions[agent_id].add(tool_name)
        
    def revoke(self, agent_id: str, tool_name: str):
        if agent_id in self.agent_permissions:
            self.agent_permissions[agent_id].discard(tool_name)
            
    def check(self, agent_id: str, tool_name: str) -> bool:
        if agent_id not in self.agent_permissions:
            return False
        return tool_name in self.agent_permissions[agent_id]

class RiskScorer:
    def __init__(self):
        self.tool_base_risk = {
            "read_file": 0.2,
            "write_file": 0.8,
            "execute_bash": 0.9,
            "calculate": 0.1,
            "python_ast": 0.5
        }
        
    def score(self, tool_name: str, args: Dict[str, Any]) -> float:
        base_risk = self.tool_base_risk.get(tool_name, 0.5)
        arg_risk = 0.0
        
        # Simple heuristic for arg risk
        for k, v in args.items():
            if isinstance(v, str):
                if "rm -rf" in v or "mkfs" in v:
                    arg_risk += 1.0
                if "/etc/passwd" in v or "/etc/shadow" in v:
                    arg_risk += 0.8
                    
        return min(1.0, 0.5 * base_risk + 0.5 * arg_risk)

class ToolRuntime:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.rate_limiter = TokenBucketRateLimiter(capacity=10, refill_rate=2)
        self.sandbox = PermissionSandbox()
        self.audit_log = ToolAuditLog()
        self.risk_scorer = RiskScorer()
        self.risk_threshold = 0.8
        
    def register_tool(self, name: str, func: Callable):
        self.tools[name] = func
        
    def execute(self, agent_id: str, tool_name: str, args: Dict[str, Any]) -> Any:
        try:
            # 1. Check Rate Limit
            if not self.rate_limiter.consume(1):
                self.audit_log.log(agent_id, tool_name, args, "RATE_LIMITED")
                raise RateLimitError(json.dumps({"error": "RateLimitExceeded", "retry_in_seconds": 1}))
                
            # 2. Check Permissions (Sandbox)
            if not self.sandbox.check(agent_id, tool_name):
                self.audit_log.log(agent_id, tool_name, args, "PERMISSION_DENIED")
                raise SecurityError(f"Agent {agent_id} lacks permission for {tool_name}")
                
            # 3. Check Risk Score
            risk = self.risk_scorer.score(tool_name, args)
            if risk >= self.risk_threshold:
                self.audit_log.log(agent_id, tool_name, args, "BLOCKED_BY_RISK_SCORE", risk)
                raise SecurityError(f"Action blocked due to high risk score: {risk}")
                
            # 4. Tool Existence
            if tool_name not in self.tools:
                self.audit_log.log(agent_id, tool_name, args, "TOOL_NOT_FOUND")
                raise ToolExecutionError(f"Tool {tool_name} not found")
                
            # 5. Execute
            func = self.tools[tool_name]
            result = func(**args)
            
            # 6. Log success
            self.audit_log.log(agent_id, tool_name, args, "SUCCESS", result)
            return result
            
        except Exception as e:
            if not isinstance(e, (RateLimitError, SecurityError, ToolExecutionError)):
                self.audit_log.log(agent_id, tool_name, args, "EXECUTION_FAILED", str(e))
                raise ToolExecutionError(str(e))
            raise

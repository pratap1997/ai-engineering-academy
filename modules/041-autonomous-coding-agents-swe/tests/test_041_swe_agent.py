import pytest
import importlib.util
import sys
import os

def load_module():
    module_name = "module_041_impl"
    file_path = os.path.join(os.path.dirname(__file__), "..", "04-implementation.py")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

mod = load_module()

# Category 1: Workspace File Operations (4 tests)
def test_workspace_initialization():
    ws = mod.WorkspaceEnvironment({"main.py": "print('hello')"})
    assert ws.list_files() == ["main.py"]

def test_workspace_read_file():
    ws = mod.WorkspaceEnvironment({"main.py": "print('hello')"})
    assert ws.read_file("main.py") == "print('hello')"

def test_workspace_read_missing_file():
    ws = mod.WorkspaceEnvironment()
    with pytest.raises(FileNotFoundError):
        ws.read_file("missing.py")

def test_workspace_write_file():
    ws = mod.WorkspaceEnvironment()
    ws.write_file("new.py", "x = 1")
    assert ws.read_file("new.py") == "x = 1"

# Category 2: Diff Patching (4 tests)
def test_diff_patcher_success():
    orig = "def add(a, b):\n    return a + c\n"
    search = "return a + c"
    replace = "return a + b"
    patched = mod.UnifiedDiffPatcher.apply_patch(orig, search, replace)
    assert "return a + b" in patched

def test_diff_patcher_not_found():
    orig = "def add(a, b):\n    return a + b\n"
    with pytest.raises(ValueError, match="not found"):
        mod.UnifiedDiffPatcher.apply_patch(orig, "return a + c", "return a + b")

def test_diff_patcher_ambiguous():
    orig = "x = 1\nx = 1\n"
    with pytest.raises(ValueError, match="ambiguous"):
        mod.UnifiedDiffPatcher.apply_patch(orig, "x = 1", "x = 2")

def test_diff_patcher_exact_match():
    orig = "hello world"
    patched = mod.UnifiedDiffPatcher.apply_patch(orig, "hello world", "goodbye world")
    assert patched == "goodbye world"

# Category 3: Syntax Validation (4 tests)
def test_syntax_valid():
    valid, err = mod.SyntaxValidator.validate("def foo():\n    pass")
    assert valid is True
    assert err is None

def test_syntax_invalid():
    valid, err = mod.SyntaxValidator.validate("def foo():\npass")
    assert valid is False
    assert "SyntaxError" in err or "IndentationError" in err

def test_syntax_invalid_char():
    valid, err = mod.SyntaxValidator.validate("x = $1")
    assert valid is False
    assert "SyntaxError" in err

def test_syntax_empty():
    valid, err = mod.SyntaxValidator.validate("")
    assert valid is True

# Category 4: Repair Loop Execution (4 tests)
def test_repair_loop_immediate_success():
    ws = mod.WorkspaceEnvironment({"main.py": "x = 1"})
    runner = mod.TestRunnerSimulator({"main.py": lambda content: (True, "Pass")})
    
    def mock_llm(history, ws):
        return "run_tests", ()
        
    agent = mod.SWEReActAgent(ws, runner, mock_llm)
    assert agent.run_repair_loop("test") is True

def test_repair_loop_one_edit_success():
    ws = mod.WorkspaceEnvironment({"main.py": "x = 1"})
    runner = mod.TestRunnerSimulator({"main.py": lambda content: ("x = 2" in content, "Failed")})
    
    def mock_llm(history, ws):
        if len(history) == 1:
            return "edit_file", ("main.py", "x = 1", "x = 2")
        return "run_tests", ()
        
    agent = mod.SWEReActAgent(ws, runner, mock_llm)
    assert agent.run_repair_loop("test") is True
    assert ws.read_file("main.py") == "x = 2"

def test_repair_loop_syntax_error_recovery():
    ws = mod.WorkspaceEnvironment({"main.py": "x = 1"})
    runner = mod.TestRunnerSimulator({"main.py": lambda content: ("x = 2" in content, "Passed" if "x = 2" in content else "Failed")})
    
    def mock_llm(history, ws):
        if len(history) == 1:
            return "edit_file", ("main.py", "x = 1", "x = $")
        elif "Syntax Error" in history[-1]:
            return "edit_file", ("main.py", "x = 1", "x = 2")
        else:
            return "run_tests", ()
        
    agent = mod.SWEReActAgent(ws, runner, mock_llm)
    assert agent.run_repair_loop("test", max_attempts=4) is True

def test_repair_loop_max_attempts_exceeded():
    ws = mod.WorkspaceEnvironment({"main.py": "x = 1"})
    runner = mod.TestRunnerSimulator({"main.py": lambda content: (False, "Failed")})
    
    def mock_llm(history, ws):
        return "run_tests", ()
        
    agent = mod.SWEReActAgent(ws, runner, mock_llm)
    assert agent.run_repair_loop("test", max_attempts=2) is False

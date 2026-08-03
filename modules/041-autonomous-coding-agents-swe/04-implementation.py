import ast
import difflib
from typing import Dict, List, Tuple, Optional, Any

class WorkspaceEnvironment:
    def __init__(self, initial_files: Dict[str, str] = None):
        self.files = initial_files or {}
    
    def read_file(self, filepath: str) -> str:
        if filepath not in self.files:
            raise FileNotFoundError(f"File {filepath} not found.")
        return self.files[filepath]
        
    def write_file(self, filepath: str, content: str):
        self.files[filepath] = content
        
    def list_files(self) -> List[str]:
        return list(self.files.keys())

class UnifiedDiffPatcher:
    @staticmethod
    def apply_patch(original_content: str, search_text: str, replace_text: str) -> str:
        if search_text not in original_content:
            raise ValueError("Search text not found in the original content.")
        if original_content.count(search_text) > 1:
            raise ValueError("Search text is ambiguous (found multiple times).")
        return original_content.replace(search_text, replace_text)

class SyntaxValidator:
    @staticmethod
    def validate(code: str) -> Tuple[bool, Optional[str]]:
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}"

class TestRunnerSimulator:
    def __init__(self, tests: Dict[str, Any]):
        self.tests = tests
        
    def run_tests(self, workspace: WorkspaceEnvironment) -> Tuple[bool, str]:
        for filename, test_func in self.tests.items():
            if filename in workspace.files:
                passed, msg = test_func(workspace.files[filename])
                if not passed:
                    return False, f"Test failed in {filename}: {msg}"
        return True, "All tests passed."

class SWEReActAgent:
    def __init__(self, workspace: WorkspaceEnvironment, test_runner: TestRunnerSimulator, mock_llm: callable):
        self.workspace = workspace
        self.test_runner = test_runner
        self.mock_llm = mock_llm
        
    def run_repair_loop(self, task: str, max_attempts: int = 3) -> bool:
        history = [f"Task: {task}"]
        
        for attempt in range(max_attempts):
            action, args = self.mock_llm(history, self.workspace)
            history.append(f"Action: {action}, Args: {args}")
            
            if action == "edit_file":
                filepath, search_txt, replace_txt = args
                content = self.workspace.read_file(filepath)
                try:
                    new_content = UnifiedDiffPatcher.apply_patch(content, search_txt, replace_txt)
                    is_valid, err = SyntaxValidator.validate(new_content)
                    if not is_valid:
                        history.append(f"Syntax Error: {err}")
                        continue
                    self.workspace.write_file(filepath, new_content)
                    history.append(f"Edited {filepath}")
                except Exception as e:
                    history.append(f"Patch Error: {e}")
                    continue
                    
            elif action == "run_tests":
                passed, msg = self.test_runner.run_tests(self.workspace)
                history.append(f"Test Result: {passed} - {msg}")
                if passed:
                    return True
            else:
                history.append(f"Unknown action: {action}")
                
        return False

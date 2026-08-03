import re
import json
import math
import random

class Tool:
    """Represents a callable tool with name, description, and function."""
    def __init__(self, name: str, description: str, fn: callable, params: dict):
        self.name = name
        self.description = description
        self.fn = fn
        self.params = params

    def execute(self, **kwargs) -> str:
        try:
            return str(self.fn(**kwargs))
        except Exception as e:
            return f"Error executing tool {self.name}: {str(e)}"

class ToolRegistry:
    """Registry for managing and selecting tools."""
    def __init__(self):
        self.tools = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found.")
        return self.tools[name]

    def list_tools(self) -> str:
        return "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])

    def execute_tool(self, name: str, **kwargs) -> str:
        return self.get(name).execute(**kwargs)

class Thought:
    """A single reasoning step in the ReAct loop."""
    def __init__(self, content: str):
        self.content = content
        
    def __repr__(self):
        return f"Thought: {self.content}"

class Action:
    """A tool call action with parsed name and arguments."""
    def __init__(self, tool_name: str, args: dict):
        self.tool_name = tool_name
        self.args = args
        
    def __repr__(self):
        return f"Action: {self.tool_name} with args {self.args}"

    @classmethod
    def parse(cls, text: str) -> 'Action':
        match = re.search(r"Action:\s*([a-zA-Z0-9_]+)\((.*)\)", text)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            try:
                args = json.loads(args_str) if args_str.strip() else {}
                return cls(tool_name, args)
            except json.JSONDecodeError:
                return cls(tool_name, {"raw_args": args_str})
        
        lines = text.strip().split('\n')
        tool_name = ""
        args_str = ""
        for line in lines:
            if line.startswith("Action:"):
                tool_name = line[len("Action:"):].strip()
            elif line.startswith("Action Input:"):
                args_str = line[len("Action Input:"):].strip()
        
        if tool_name:
            try:
                args = json.loads(args_str) if args_str else {}
                return cls(tool_name, args)
            except json.JSONDecodeError:
                return cls(tool_name, {"input": args_str})
                
        raise ValueError("Could not parse action from text: " + text)

class Observation:
    """Result from executing an action."""
    def __init__(self, content: str, success: bool):
        self.content = content
        self.success = success
        
    def __repr__(self):
        return f"Observation: {self.content}"

class TrajectoryStep:
    """One step in the agent trajectory: thought + action + observation."""
    def __init__(self, thought: Thought, action: Action, observation: Observation):
        self.thought = thought
        self.action = action
        self.observation = observation

class ReActAgent:
    """ReAct agent implementing the Reasoning+Acting loop."""
    
    def __init__(self, llm_fn: callable, tools: list[Tool], max_steps: int = 10):
        self.llm_fn = llm_fn
        self.registry = ToolRegistry()
        for t in tools:
            self.registry.register(t)
        self.max_steps = max_steps
    
    def _build_prompt(self, task: str, trajectory: list[TrajectoryStep]) -> str:
        """Build the ReAct prompt with task + trajectory history."""
        prompt = f"Task: {task}\n\nAvailable Tools:\n"
        prompt += self.registry.list_tools() + "\n\n"
        
        prompt += "To use a tool, format your output as:\n"
        prompt += "Thought: <your reasoning>\n"
        prompt += "Action: <tool_name>\n"
        prompt += "Action Input: <json dict of arguments>\n\n"
        prompt += "If you have the final answer, format it as:\n"
        prompt += "Thought: <reasoning>\n"
        prompt += "Action: FINISH\n"
        prompt += "Action Input: {\"answer\": \"<your answer>\"}\n\n"
        
        prompt += "Begin!\n\n"
        
        for i, step in enumerate(trajectory):
            prompt += f"Step {i+1}:\n"
            prompt += f"{step.thought}\n"
            prompt += f"Action: {step.action.tool_name}\n"
            prompt += f"Action Input: {json.dumps(step.action.args)}\n"
            prompt += f"{step.observation}\n\n"
            
        return prompt
    
    def _parse_llm_output(self, output: str) -> tuple[Thought, Action | None]:
        """Parse LLM output into thought + optional action."""
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", output, re.DOTALL)
        thought_content = thought_match.group(1).strip() if thought_match else "No thought provided."
        thought = Thought(thought_content)
        
        try:
            action = Action.parse(output)
            return thought, action
        except ValueError:
            return thought, None
    
    def run(self, task: str) -> dict:
        """Run the full ReAct loop until FINISH or max_steps."""
        trajectory = []
        answer = None
        success = False
        
        for step_num in range(self.max_steps):
            prompt = self._build_prompt(task, trajectory)
            llm_response = self.llm_fn(prompt)
            thought, action = self._parse_llm_output(llm_response)
            
            if not action:
                obs = Observation("Error: Invalid action format. Please provide Action and Action Input.", False)
                trajectory.append(TrajectoryStep(thought, Action("ERROR", {}), obs))
                continue
                
            if action.tool_name == "FINISH":
                answer = action.args.get("answer", str(action.args))
                success = True
                trajectory.append(TrajectoryStep(thought, action, Observation("Task completed.", True)))
                break
                
            try:
                result = self.registry.execute_tool(action.tool_name, **action.args)
                obs = Observation(result, True)
            except Exception as e:
                obs = Observation(f"Tool Execution Error: {str(e)}", False)
                
            trajectory.append(TrajectoryStep(thought, action, obs))
            
        return {
            "answer": answer if answer else "Max steps reached without finishing.",
            "trajectory": trajectory,
            "steps": len(trajectory),
            "success": success
        }

class TaskDecomposer:
    """Decomposes complex tasks into ordered subtasks."""
    def __init__(self, llm_fn: callable):
        self.llm_fn = llm_fn
        
    def decompose(self, task: str) -> list[str]:
        prompt = f"Decompose the following task into a list of simpler subtasks. Output one subtask per line.\nTask: {task}\nSubtasks:\n"
        response = self.llm_fn(prompt)
        subtasks = [line.strip().lstrip("-*1234567890. ") for line in response.split('\n') if line.strip()]
        return subtasks

class UCBToolSelector:
    """Upper Confidence Bound tool selector (bandit algorithm)."""
    def __init__(self, tools: list[str], c: float = 1.414):
        self.tools = tools
        self.c = c
        self.counts = {t: 0 for t in tools}
        self.values = {t: 0.0 for t in tools}
        self.total_pulls = 0
        
    def select(self) -> str:
        for t in self.tools:
            if self.counts[t] == 0:
                return t
                
        best_tool = None
        best_score = -float('inf')
        
        for t in self.tools:
            exploitation = self.values[t]
            exploration = self.c * math.sqrt(math.log(self.total_pulls) / self.counts[t])
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_tool = t
                
        return best_tool
        
    def update(self, tool_name: str, reward: float) -> None:
        self.counts[tool_name] += 1
        self.total_pulls += 1
        n = self.counts[tool_name]
        value = self.values[tool_name]
        new_value = ((n - 1) * value + reward) / n
        self.values[tool_name] = new_value
        
    def ucb_scores(self) -> dict[str, float]:
        scores = {}
        for t in self.tools:
            if self.counts[t] == 0:
                scores[t] = float('inf')
            else:
                exploitation = self.values[t]
                exploration = self.c * math.sqrt(math.log(self.total_pulls) / self.counts[t])
                scores[t] = exploitation + exploration
        return scores

def make_calculator_tool() -> Tool:
    def calc(expression: str) -> str:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression."
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"
            
    return Tool(
        name="Calculator",
        description="Evaluates mathematical expressions.",
        fn=calc,
        params={"expression": "string"}
    )

def make_search_tool(knowledge_base: dict) -> Tool:
    def search(query: str) -> str:
        query_lower = query.lower()
        for key, value in knowledge_base.items():
            if query_lower in key.lower():
                return value
        return "No information found."
        
    return Tool(
        name="Search",
        description="Searches a knowledge base for a query.",
        fn=search,
        params={"query": "string"}
    )

def make_python_executor_tool() -> Tool:
    def execute_python(code: str) -> str:
        import io
        import sys
        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()
        try:
            exec(code, {})
            sys.stdout = old_stdout
            return redirected_output.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            return f"Error: {e}"
            
    return Tool(
        name="PythonExecutor",
        description="Executes python code and returns stdout.",
        fn=execute_python,
        params={"code": "string"}
    )

def make_text_processor_tool() -> Tool:
    def process_text(text: str, operation: str) -> str:
        if operation == "uppercase":
            return text.upper()
        elif operation == "lowercase":
            return text.lower()
        elif operation == "count":
            return str(len(text))
        return "Error: Unknown operation."
        
    return Tool(
        name="TextProcessor",
        description="Processes text. Operations: uppercase, lowercase, count.",
        fn=process_text,
        params={"text": "string", "operation": "string"}
    )

if __name__ == '__main__':
    def mock_llm(prompt: str) -> str:
        if "What is 25 * 4 plus the age of the universe in billions of years?" in prompt:
            if "Observation:" not in prompt:
                return '''Thought: I need to calculate 25 * 4 first, then find the age of the universe.
Action: Calculator
Action Input: {"expression": "25 * 4"}'''
            elif "100" in prompt and "Search" not in prompt:
                return '''Thought: The result is 100. Now I need to find the age of the universe.
Action: Search
Action Input: {"query": "age of the universe"}'''
            elif "13.8" in prompt:
                return '''Thought: The age of the universe is 13.8 billion years. I need to add 100 + 13.8.
Action: Calculator
Action Input: {"expression": "100 + 13.8"}'''
            elif "113.8" in prompt:
                return '''Thought: The final answer is 113.8.
Action: FINISH
Action Input: {"answer": "113.8"}'''
        
        return '''Thought: I don't know what to do.
Action: FINISH
Action Input: {"answer": "Failed."}'''

    kb = {"age of the universe": "The universe is 13.8 billion years old."}
    tools = [make_calculator_tool(), make_search_tool(kb)]
    agent = ReActAgent(mock_llm, tools)
    
    task = "What is 25 * 4 plus the age of the universe in billions of years?"
    print(f"Task: {task}\\n")
    
    result = agent.run(task)
    
    for i, step in enumerate(result['trajectory']):
        print(f"--- Step {i+1} ---")
        print(step.thought)
        print(step.action)
        print(step.observation)
        print()
        
    print(f"Final Answer: {result['answer']}")
    print(f"Success: {result['success']} in {result['steps']} steps.")

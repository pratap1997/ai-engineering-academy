import json
import re
from typing import Any, Callable

class JSONSchema:
    """Represents a JSON schema with validation."""
    def __init__(self, schema: dict):
        self.schema = schema

    def validate(self, data: dict) -> tuple[bool, list[str]]:
        errors = []
        if not isinstance(data, dict):
            return False, ["Root data must be a dictionary/object."]
            
        req_fields = self.required_fields()
        for req in req_fields:
            if req not in data:
                errors.append(f"Missing required field: '{req}'")
                
        types = self.field_types()
        for key, value in data.items():
            if key in types:
                expected_type = types[key]
                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Field '{key}' should be string, got {type(value).__name__}")
                elif expected_type == "integer" and not isinstance(value, int):
                    errors.append(f"Field '{key}' should be integer, got {type(value).__name__}")
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Field '{key}' should be number, got {type(value).__name__}")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Field '{key}' should be boolean, got {type(value).__name__}")
                elif expected_type == "array" and not isinstance(value, list):
                    errors.append(f"Field '{key}' should be array, got {type(value).__name__}")
                elif expected_type == "object" and not isinstance(value, dict):
                    errors.append(f"Field '{key}' should be object, got {type(value).__name__}")

        return len(errors) == 0, errors

    def required_fields(self) -> list[str]:
        return self.schema.get("required", [])

    def field_types(self) -> dict[str, str]:
        props = self.schema.get("properties", {})
        return {k: v.get("type", "any") for k, v in props.items()}

class StructuredOutput:
    """Wrapper around LLM output that enforces a schema."""
    def __init__(self, schema: JSONSchema, data: dict, raw_text: str):
        self.schema = schema
        self.data = data
        self.raw_text = raw_text
        self._is_valid, self._errors = self.schema.validate(self.data)

    def is_valid(self) -> bool:
        return self._is_valid

    def get(self, field: str, default=None) -> Any:
        return self.data.get(field, default)

class JSONExtractor:
    """Extracts JSON from raw LLM output."""
    @staticmethod
    def extract(text: str) -> dict | None:
        # First fix common errors
        text = JSONExtractor.fix_common_errors(text)
        
        # Try finding json blocks
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback to finding outermost curly braces
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                return None
                
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def fix_common_errors(text: str) -> str:
        # Fix trailing commas before closing braces
        text = re.sub(r',\s*([\]}])', r'\1', text)
        return text

class ToolSchema:
    """Defines a callable tool with parameter schema."""
    def __init__(self, name: str, description: str, parameters: dict, required: list[str]):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.required = required

    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required
                }
            }
        }

    def to_anthropic_format(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required
            }
        }

    def validate_call(self, args: dict) -> tuple[bool, list[str]]:
        schema_dict = {
            "type": "object",
            "properties": self.parameters,
            "required": self.required
        }
        schema = JSONSchema(schema_dict)
        return schema.validate(args)

class ToolCall:
    """Represents a parsed function/tool call from LLM output."""
    def __init__(self, tool_name: str, args: dict):
        self.tool_name = tool_name
        self.args = args

    @classmethod
    def parse_openai_format(cls, response: dict) -> 'ToolCall':
        if "tool_calls" in response and len(response["tool_calls"]) > 0:
            call = response["tool_calls"][0]["function"]
            args = json.loads(call["arguments"])
            return cls(call["name"], args)
        return cls("", {})

    @classmethod
    def parse_anthropic_format(cls, response: dict) -> 'ToolCall':
        # Simulated format for Anthropic tool use
        if "content" in response:
            for item in response["content"]:
                if item.get("type") == "tool_use":
                    return cls(item["name"], item["input"])
        return cls("", {})

class StructuredLLM:
    """LLM wrapper that enforces structured outputs with retry logic."""
    def __init__(self, llm_fn: Callable[[str], str], max_retries: int = 3):
        self.llm_fn = llm_fn
        self.max_retries = max_retries

    def call_with_schema(self, prompt: str, schema: JSONSchema) -> StructuredOutput:
        for attempt in range(self.max_retries):
            raw_response = self.llm_fn(prompt)
            data = JSONExtractor.extract(raw_response)
            
            if data is not None:
                output = StructuredOutput(schema, data, raw_response)
                if output.is_valid():
                    return output
            
            # Simulated retry prompt could be appended here in a real scenario
            prompt += "\n\nPlease ensure you output VALID JSON conforming strictly to the schema."
            
        return StructuredOutput(schema, {}, "Max retries exceeded")

    def call_with_tool(self, prompt: str, tools: list[ToolSchema]) -> ToolCall:
        # Simplified simulation: we assume the llm_fn returns a JSON string simulating OpenAI's response format
        raw_response = self.llm_fn(prompt)
        try:
            data = json.loads(raw_response)
            return ToolCall.parse_openai_format(data)
        except json.JSONDecodeError:
            return ToolCall("", {})

class ConstrainedDecoder:
    """Simulates grammar-constrained decoding (from-scratch implementation)."""
    def __init__(self, schema: JSONSchema):
        self.schema = schema

    def is_valid_prefix(self, partial_json: str) -> bool:
        # Simplified simulation: checks if brackets are balanced
        stack = []
        in_string = False
        escape = False
        
        for char in partial_json:
            if escape:
                escape = False
                continue
                
            if char == '\\':
                escape = True
            elif char == '"':
                in_string = not in_string
            elif not in_string:
                if char in '{[':
                    stack.append(char)
                elif char in '}]':
                    if not stack:
                        return False
                    last = stack.pop()
                    if (char == '}' and last != '{') or (char == ']' and last != '['):
                        return False
        return True

    def valid_next_tokens(self, partial_json: str, vocab: list[str]) -> list[str]:
        # Filter tokens that would violate the prefix check
        return [t for t in vocab if self.is_valid_prefix(partial_json + t)]

    def generate(self, llm_fn: Callable[[str], str], prompt: str) -> dict:
        # Very simplified simulation of token-by-token generation
        res = llm_fn(prompt)
        return json.loads(res) if res else {}

if __name__ == "__main__":
    # Demo
    schema = JSONSchema({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    })
    
    # Simulate a messy response
    def mock_llm(prompt):
        return "Here is the result:\n```json\n{\n  \"name\": \"Alice\",\n  \"age\": 30,\n}\n```\nHope it helps!"
        
    s_llm = StructuredLLM(mock_llm)
    out = s_llm.call_with_schema("Get Alice, 30", schema)
    print(f"Valid: {out.is_valid()}, Name: {out.get('name')}, Age: {out.get('age')}")

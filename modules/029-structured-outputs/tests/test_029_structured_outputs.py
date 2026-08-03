import pytest
import os
import sys
import importlib.util

# Load the implementation module dynamically since it starts with a number
module_name = 'module_029_impl'
file_path = os.path.join(os.path.dirname(__file__), '..', '04-implementation.py')
spec = importlib.util.spec_from_file_location(module_name, file_path)
impl = importlib.util.module_from_spec(spec)
sys.modules[module_name] = impl
spec.loader.exec_module(impl)

JSONSchema = impl.JSONSchema
JSONExtractor = impl.JSONExtractor
ToolSchema = impl.ToolSchema
ToolCall = impl.ToolCall
StructuredLLM = impl.StructuredLLM
ConstrainedDecoder = impl.ConstrainedDecoder

# ==========================================
# Category 1: JSON Schema (4)
# ==========================================

@pytest.fixture
def person_schema():
    return JSONSchema({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    })

def test_schema_validates_correct_data(person_schema):
    valid, errors = person_schema.validate({"name": "Alice", "age": 30})
    assert valid is True
    assert len(errors) == 0

def test_schema_rejects_missing_required_field(person_schema):
    valid, errors = person_schema.validate({"age": 30})
    assert valid is False
    assert any("name" in e for e in errors)

def test_schema_rejects_wrong_type(person_schema):
    valid, errors = person_schema.validate({"name": "Alice", "age": "thirty"})
    assert valid is False
    assert any("integer" in e for e in errors)

def test_schema_required_fields_list(person_schema):
    req = person_schema.required_fields()
    assert req == ["name"]

# ==========================================
# Category 2: JSON Extraction (4)
# ==========================================

def test_extractor_handles_clean_json():
    text = '{"key": "value"}'
    res = JSONExtractor.extract(text)
    assert res == {"key": "value"}

def test_extractor_handles_markdown_code_block():
    text = 'Here is the JSON:\n```json\n{"key": "value"}\n```\nDone.'
    res = JSONExtractor.extract(text)
    assert res == {"key": "value"}

def test_extractor_fixes_trailing_comma():
    text = '{"key": "value",}'
    res = JSONExtractor.extract(text)
    assert res == {"key": "value"}

def test_extractor_returns_none_on_invalid():
    text = 'This is not json.'
    res = JSONExtractor.extract(text)
    assert res is None

# ==========================================
# Category 3: Tool Schema (4)
# ==========================================

@pytest.fixture
def weather_tool():
    return ToolSchema(
        name="get_weather",
        description="Get current weather",
        parameters={"location": {"type": "string"}},
        required=["location"]
    )

def test_tool_schema_to_openai_format(weather_tool):
    fmt = weather_tool.to_openai_format()
    assert fmt["type"] == "function"
    assert fmt["function"]["name"] == "get_weather"
    assert fmt["function"]["parameters"]["required"] == ["location"]

def test_tool_schema_validates_valid_call(weather_tool):
    valid, _ = weather_tool.validate_call({"location": "London"})
    assert valid is True

def test_tool_schema_rejects_invalid_call(weather_tool):
    valid, _ = weather_tool.validate_call({"city": "London"})
    assert valid is False

def test_tool_call_parse_openai_format():
    response = {
        "tool_calls": [{
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Paris"}'
            }
        }]
    }
    call = ToolCall.parse_openai_format(response)
    assert call.tool_name == "get_weather"
    assert call.args == {"location": "Paris"}

# ==========================================
# Category 4: Structured LLM (4)
# ==========================================

def test_structured_llm_returns_valid_output(person_schema):
    def mock_llm(prompt):
        return '{"name": "Bob", "age": 40}'
    
    llm = StructuredLLM(mock_llm, max_retries=1)
    out = llm.call_with_schema("prompt", person_schema)
    assert out.is_valid() is True
    assert out.get("name") == "Bob"

def test_structured_llm_retries_on_invalid(person_schema):
    call_count = 0
    def mock_llm(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '{"age": 40}' # invalid
        return '{"name": "Bob", "age": 40}'
    
    llm = StructuredLLM(mock_llm, max_retries=2)
    out = llm.call_with_schema("prompt", person_schema)
    assert out.is_valid() is True
    assert call_count == 2

def test_structured_llm_max_retries_exceeded(person_schema):
    def mock_llm(prompt):
        return 'invalid'
    
    llm = StructuredLLM(mock_llm, max_retries=2)
    out = llm.call_with_schema("prompt", person_schema)
    assert out.is_valid() is False
    assert out.raw_text == "Max retries exceeded"

def test_constrained_decoder_valid_prefix():
    decoder = ConstrainedDecoder(JSONSchema({}))
    assert decoder.is_valid_prefix('{"a": 1}') is True
    assert decoder.is_valid_prefix('{"a": ') is True
    assert decoder.is_valid_prefix('{"a": 1]') is False

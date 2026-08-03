from implementation import JSONExtractor, JSONSchema, ToolSchema, StructuredLLM

def experiment_1_extraction_reliability():
    print("--- Experiment 1: Extraction Reliability ---")
    texts = [
        '{"name": "Alice"}',
        'Here is the JSON:\n```\n{"name": "Bob"}\n```',
        '{"name": "Charlie",}', # trailing comma
        'No json here.'
    ]
    for i, t in enumerate(texts):
        res = JSONExtractor.extract(t)
        print(f"Test {i+1}: {'Success' if res else 'Failed'} -> {res}")
    print()

def experiment_2_schema_validation():
    print("--- Experiment 2: Schema Validation ---")
    schema = JSONSchema({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "score": {"type": "number"}
        },
        "required": ["name"]
    })
    
    test_cases = [
        ({"name": "Dave", "score": 95.5}, True),
        ({"name": "Eve"}, True),
        ({"score": 100}, False), # missing name
        ({"name": "Frank", "score": "high"}, False) # wrong type
    ]
    
    for i, (data, expected) in enumerate(test_cases):
        valid, errors = schema.validate(data)
        success = valid == expected
        print(f"Test {i+1}: {'Pass' if success else 'Fail'} (Expected {expected}, Got {valid}, Errors: {errors})")
    print()

def experiment_3_retry_effectiveness():
    print("--- Experiment 3: Retry Effectiveness ---")
    
    call_count = 0
    def mock_flaky_llm(prompt):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return "{\"name\": \"Grace\",}" # invalid (trailing comma handled by extractor though)
        return "{\"name\": \"Grace\", \"age\": 25}"
        
    schema = JSONSchema({
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"]
    })
    
    llm = StructuredLLM(mock_flaky_llm, max_retries=3)
    out = llm.call_with_schema("Test", schema)
    print(f"Extraction valid: {out.is_valid()}, Retries used: {call_count}")
    print()

def experiment_4_tool_selection():
    print("--- Experiment 4: Tool Selection ---")
    tool = ToolSchema(
        name="get_weather",
        description="Get weather for a location",
        parameters={"location": {"type": "string"}},
        required=["location"]
    )
    
    print("OpenAI format:", tool.to_openai_format())
    print("Anthropic format:", tool.to_anthropic_format())
    print()

if __name__ == "__main__":
    experiment_1_extraction_reliability()
    experiment_2_schema_validation()
    experiment_3_retry_effectiveness()
    experiment_4_tool_selection()

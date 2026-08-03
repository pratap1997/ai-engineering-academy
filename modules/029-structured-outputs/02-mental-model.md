# 02 - Mental Model: Forms, Trees, and Filters

## The "Fill in the Form" Metaphor
Think of an LLM as a highly capable but highly unconstrained worker. If you ask them to collect data on a person, they might write a paragraph, an essay, or a bulleted list. 

A **JSON Schema** is like handing that worker a strict **form template**. The LLM becomes a form-filler. The schema defines the required fields (boxes to fill out) and the expected types (numbers, text, checkboxes). The LLM's job shifts from "write text" to "populate this specific data structure."

## JSON Schema as a Constraint Tree
When an LLM generates structured output, it navigates a **constraint tree**. 
- The root of the tree is the opening brace `{`.
- The branches represent possible keys (e.g., `"name"`, `"age"`).
- The leaves represent valid values based on the schema (strings, integers).

If the schema requires a boolean, the constraint tree prunes all branches except `true` and `false`. The LLM can only generate valid leaves.

## Function Calling as "Structured Intent Detection"
When using **function calling** or **tools**, the mental model shifts slightly.
1. **Intent Detection**: The LLM reads the user prompt and decides *which* function to call (e.g., `get_weather` vs `search_web`).
2. **Argument Generation**: The LLM then acts as a form-filler for that specific function's schema.

It's essentially classification followed by schema-constrained extraction.

## Constrained Decoding: Pruning the Token Tree
In standard text generation, the LLM assigns probabilities to all 50,000+ tokens in its vocabulary. 
In **constrained decoding**, we apply a hard filter over the token tree at every step.

```text
Prompt: "Extract: John is 25"
Schema: {"name": str, "age": int}

Generation Step 1:
Valid next tokens: "{"
LLM probabilities: P("{") = 1.0 (forced)

Generation Step 2:
Valid next tokens: "\"name\": \""
LLM generates: "John"

Generation Step 3:
Valid next tokens: "\", \"age\":"
LLM generates: "25"

Generation Step 4:
Valid next tokens: "}"
LLM probabilities: P("}") = 1.0 (forced)
```

By pruning invalid tokens before sampling, we force the model down a path that *must* result in valid JSON.

## The Reliability Spectrum
Understanding when to trust your outputs involves knowing where your method sits on the reliability spectrum:

1. **Unstructured (0% Guarantee)**: Standard prompt. Often breaks downstream systems.
2. **JSON Mode (80% Guarantee)**: API feature that ensures the output parses as JSON, but doesn't guarantee the schema structure.
3. **Schema-Constrained (95%+ Guarantee)**: Pydantic parsing with strict instructions or OpenAI's structured outputs.
4. **Grammar-Constrained (100% Guarantee)**: Token-level masking (e.g., Outlines). Mathematically impossible to produce invalid structures.

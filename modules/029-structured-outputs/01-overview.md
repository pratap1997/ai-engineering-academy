# 01 - Overview: Structured Outputs & Function Calling

## The Reliability Problem
Large Language Models (LLMs) naturally produce free-form text. They are autocomplete engines trained to generate a sequence of tokens that is probable given a context. However, most software applications require structured data—like JSON—to interact reliably with APIs, databases, and control flows. 

If an LLM is asked to extract a user's details, returning `"Here is the information you requested: {\"name\": \"Alice\", \"age\": 30}"` breaks systems that expect pure JSON. We need ways to guarantee that the LLM's output conforms strictly to an expected structure.

## Three Approaches to Structured Output

### 1. Prompt-Based (Instruction to Output JSON)
This is the simplest but least reliable method. You instruct the LLM in the system prompt to "Output ONLY valid JSON" or "Reply in the following JSON format...". 
* **Pros**: Works on any model without special API features.
* **Cons**: Low reliability. Models often include conversational filler ("Here is the JSON:"), miss required fields, or produce syntax errors like trailing commas.

### 2. Schema-Constrained (e.g., Pydantic + Response Format)
APIs like OpenAI and Anthropic provide native support for structured outputs.
* **OpenAI**: Using `response_format: {type: "json_schema"}` forces the model to adhere to a provided JSON Schema.
* **Anthropic**: The `tool_use` pattern forces the model to emit a structured payload corresponding to a tool's arguments.
* **Pros**: Much higher reliability (usually >95%). Handled server-side.
* **Cons**: Requires specific API support. Still occasionally hallucinates schema variations if not strictly constrained at the inference engine level.

### 3. Constrained Decoding (Grammar-Based)
This is the most mathematically rigorous approach. Libraries like **Outlines** use the schema to modify the LLM's decoding process in real-time. If the next character must be a quote `"` to satisfy JSON syntax, the probabilities of all other tokens are forced to zero.
* **Pros**: 100% guarantee of schema compliance (structural).
* **Cons**: Requires access to the model's logits (open-source models or specific endpoints). More compute-intensive during generation.

## When to use structured outputs vs free-form text
* **Free-form text**: Creative writing, open-ended brainstorming, summarizing documents for human reading, general chat.
* **Structured outputs**: Data extraction, API routing, multi-agent communication, autonomous workflows, database insertion.

## Prerequisites
To fully understand this module, you should be familiar with:
* **Module 011 (Transformers)**: Understanding how logits and token probabilities work.
* **Module 026 (Agentic Loops)**: Understanding why agents need structured tools to interact with environments.

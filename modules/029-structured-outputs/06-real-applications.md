# 06 - Real Applications of Structured Outputs

## 1. OpenAI Function Calling in Production
In banking chatbots, LLMs extract customer data at scale. Instead of parsing conversational text for details like "transfer 500 dollars to checking", production systems use OpenAI's function calling. The LLM is provided a tool:
```json
{
  "name": "initiate_transfer",
  "parameters": {
    "type": "object",
    "properties": {
      "amount": {"type": "number"},
      "destination_account": {"type": "string"}
    }
  }
}
```
This guarantees the banking backend receives a predictable JSON object it can securely process.

## 2. Anthropic Tool Use for Agent Pipelines
Autonomous coding agents (like Claude working in an IDE) use the `tool_use` pattern to interact with the file system.
The agent decides between tools like `read_file`, `write_to_file`, and `run_command`. The structured output guarantees that the file path and code contents are cleanly separated from the agent's internal reasoning/thoughts.

## 3. Outlines Library in Healthcare
In medical systems, parsing errors can be dangerous. The **Outlines** library uses grammar-constrained decoding to force open-source models (like Llama 3) to fill out medical forms with 100% syntactic reliability. By modifying the logits at inference time, it ensures the model cannot output malformed JSON, even when extracting complex nested medical codes and diagnoses.

## 4. Information Extraction Pipelines (NER)
Legal firms use structured outputs for Named Entity Recognition (NER) on massive document corpora. By defining a Pydantic schema for `ContractParties` (buyer, seller, dates, clauses) and forcing the LLM to output this schema, they can convert unstructured PDFs into queryable relational databases without writing complex regex or training custom spaCy models.

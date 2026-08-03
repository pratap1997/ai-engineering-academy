# 07 - Engineering Challenge: Build a Reliable Extraction Pipeline

## Challenge Description
Your goal is to build an extraction pipeline that can pull structured data from unstructured text with >95% reliability, handling messy data and common LLM failure modes.

### The Task
You must build a pipeline to extract a `PersonRecord` from 50 different text formats. The texts range from clean formatting to very dirty, multilingual paragraphs.

### Schema Requirements
You need to extract the following fields into JSON:
- `name` (string, required)
- `age` (integer, optional)
- `email` (string, required, must be valid format)
- `tags` (list of strings, optional)

### Failure Modes to Handle
Your pipeline must gracefully handle:
1. Markdown code blocks wrapping the JSON (` ```json ... ``` `)
2. Single quotes instead of double quotes (invalid JSON)
3. Trailing commas (invalid JSON)
4. Missing required fields
5. Extraneous conversational text before or after the JSON

### Success Criteria
1. Extract the `PersonRecord` from 50 varied texts.
2. Achieve <5% failure rate (at least 48/50 successful extractions).
3. Implement a schema version migration function (transforming a V1 extraction to a V2 schema).

**No Hints.** Build the pipeline using raw Python. Do not use Pydantic or LangChain for this challenge.

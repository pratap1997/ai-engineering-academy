# 08 - Assessment

## Conceptual Questions
1. **Why do standard LLMs struggle to output reliable JSON without special constraints?**
   *Answer*: LLMs are autocomplete engines trained on free-form text. They are incentivized to produce conversational filler or Markdown, and they do not have an inherent rigorous understanding of JSON syntax without explicit constraints.

2. **What is the difference between schema-constrained API calls (like OpenAI's structured outputs) and grammar-constrained decoding (like Outlines)?**
   *Answer*: API features often use a mix of fine-tuning and server-side tricks to encourage the right format, but still sample from the whole vocabulary. Grammar-constrained decoding mathematically masks the probabilities of invalid tokens during the generation step, making invalid output impossible.

3. **In the "Fill in the form" mental model, what does the constraint tree represent?**
   *Answer*: It represents the valid paths the LLM can take. The schema restricts the branches to specific keys and value types, pruning invalid text generations.

## Mathematical Questions
4. **If an LLM has a vocabulary of 50,000 tokens, but the grammar constraint only allows a single token (e.g., `"}`), what happens to the probabilities of the other 49,999 tokens?**
   *Answer*: Their probabilities are multiplied by 0 (masked), and the remaining token's probability becomes 1.0 (100%).

5. **How does applying a JSON schema constraint affect the conditional entropy of the LLM's next-token distribution?**
   *Answer*: It drastically reduces the conditional entropy by restricting the valid vocabulary $V_{valid}$, effectively removing bits of uncertainty.

6. **Why is a Pushdown Automaton (PDA) required to validate JSON, rather than a simpler Finite State Machine (FSM)?**
   *Answer*: Because JSON allows for infinitely nested structures (objects within arrays within objects). A PDA uses a stack to keep track of open/closed brackets, which an FSM cannot do.

## Implementation Questions
7. **Write a Python regex that can extract a JSON object embedded inside markdown code blocks and conversational text.**
   *Answer*: `import re; match = re.search(r'\{.*\}', text, re.DOTALL); json_str = match.group(0) if match else None`

8. **How would you implement a simple auto-fixer for trailing commas in a JSON string before passing it to `json.loads()`?**
   *Answer*: `import re; fixed = re.sub(r',\s*([\]}])', r'\1', raw_json_string)`

## Judgment Questions
9. **You are building an AI agent that writes code files. Should you use free-form text or structured tool calling? Why?**
   *Answer*: Structured tool calling. The agent needs to definitively separate its reasoning from the exact file path and the code string. Free-form text requires brittle string parsing to extract the code.

10. **Your app uses an open-source model locally and requires extracting dates. Should you fine-tune the model on JSON data or use constrained decoding?**
    *Answer*: Constrained decoding. Fine-tuning improves the likelihood of correct JSON but cannot guarantee it 100%. Constrained decoding guarantees the structure without needing extra training data.

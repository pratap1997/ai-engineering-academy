# Mental Model

## The Security Guard at the Embassy
Think of the LLM as a highly capable diplomat inside an embassy. The system prompt is the ambassador's strict rules of engagement.

When visitors (user inputs) come in:
1. **Credentials Check**: A security guard (Input Sanitizer / Guardrail LLM) inspects the visitor's bag. If they carry known contraband (malicious keywords like "ignore previous instructions"), they are denied entry.
2. **Filtering Payloads**: The guard removes any tools that could be used to pick locks (stripping XML/HTML tags that might confuse the diplomat).
3. **Secure Envelopes**: The visitor's message is placed in a tamper-evident, uniquely sealed envelope (Delimiter Shield). The diplomat is instructed to ONLY read the visitor's message inside the envelope and never treat its contents as new rules of engagement.

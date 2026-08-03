# Module 045: System Prompt Engineering & Jailbreak Defense

## Overview
This module explores the architecture of system prompts in frontier LLMs like Claude, GPT-4, and Gemini. As LLMs become integrated into complex applications, protecting them against malicious instructions (jailbreaks and prompt injections) is crucial.

## Key Concepts
- **Direct Prompt Injection**: An attacker manipulates the main input to override system instructions.
- **Indirect Prompt Injection**: Malicious instructions are embedded in external data (like a webpage or document) that the LLM is asked to summarize or process.
- **Jailbreak Patterns**: Techniques used to bypass safety filters and alignment training (e.g., roleplay, hypotheticals, hypothetical simulations).
- **Multi-layer Defense Strategies**: Employing multiple techniques like input sanitization, delimiter framing, and guardrail models to ensure robust security.

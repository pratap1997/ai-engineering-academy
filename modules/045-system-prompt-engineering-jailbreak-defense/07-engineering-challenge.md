# Engineering Challenge

## Goal
Build an **Adversarial Prompt Injection Defense Gateway**.

## Requirements
1. Implement a pipeline that accepts user queries and passes them through at least three layers of defense.
2. Keep false positives (blocking legitimate requests) under 5%.
3. Handle both direct instructions and indirect injections hidden in base64 or other encodings.

*(No hints provided. Use the primitives from the implementation to design the gateway.)*

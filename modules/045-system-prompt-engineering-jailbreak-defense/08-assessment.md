# Assessment

1. **What is the difference between direct and indirect prompt injection?**
   *Direct injection modifies the main prompt; indirect hides the payload in external data.*
2. **Why use random nonces in delimiters?**
   *To prevent an attacker from closing the delimiter early.*
3. **What is a Dual-LLM Guardrail?**
   *A system where a separate, smaller LLM evaluates the safety of an input before passing it to the main LLM.*
4. **How do delimiters protect against prompt injection?**
   *By strictly framing untrusted inputs so they are not interpreted as instructions.*
5. **What is input sanitization?**
   *The process of filtering out or modifying inputs that contain dangerous keywords or patterns.*
6. **What is the collision probability in a 16-character alphanumeric delimiter?**
   *(1/62)^16.*
7. **What is the risk of using a fixed delimiter?**
   *An attacker can simply close the delimiter and append their own instructions.*
8. **How does an attacker use roleplay to jailbreak a model?**
   *By asking the model to assume a persona that ignores its original rules.*
9. **What are XML tags often used for in system prompts?**
   *To separate and structure instructions, rules, and user inputs clearly.*
10. **Why use a smaller LLM for the guardrail?**
   *For lower latency and cost compared to the main LLM, while still providing robust evaluation.*

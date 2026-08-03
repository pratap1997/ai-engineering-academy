import json
import re
import secrets
import string

class SystemPromptBuilder:
    def __init__(self, core_instructions):
        self.core_instructions = core_instructions
        self.rules = []
        
    def add_rule(self, rule):
        self.rules.append(rule)
        
    def build(self, user_input, delimiter=""):
        rules_str = "\n".join(f"- {r}" for r in self.rules)
        prompt = f"<system_instructions>\n{self.core_instructions}\n<rules>\n{rules_str}\n</rules>\n</system_instructions>\n"
        if delimiter:
            prompt += f"<user_input>\n{delimiter}\n{user_input}\n{delimiter}\n</user_input>"
        else:
            prompt += f"<user_input>\n{user_input}\n</user_input>"
        return prompt

class PromptSanitizer:
    def __init__(self, blocked_words=None):
        self.blocked_words = blocked_words or ["ignore previous instructions", "system prompt", "you are now", "forget"]
        
    def sanitize(self, text):
        lower_text = text.lower()
        for word in self.blocked_words:
            if word in lower_text:
                raise ValueError(f"Blocked keyword detected: {word}")
        # Strip potentially malicious XML-like tags that could confuse the parser
        sanitized = re.sub(r'<(system|user|rules)[^>]*>', '', text, flags=re.IGNORECASE)
        return sanitized

class DelimiterShield:
    def __init__(self, length=16):
        self.length = length
        
    def generate_nonce(self):
        chars = string.ascii_letters + string.digits
        return "".join(secrets.choice(chars) for _ in range(self.length))
        
    def shield_input(self, user_input):
        nonce = self.generate_nonce()
        delimiter = f"===UNTRUSTED_INPUT_{nonce}==="
        if delimiter in user_input:
            return self.shield_input(user_input)
        return delimiter

class DualLLMGuardrail:
    def __init__(self, safe_threshold=0.8):
        self.safe_threshold = safe_threshold
        
    def mock_evaluator_llm(self, input_text):
        lower_text = input_text.lower()
        danger_signals = ["ignore", "override", "bypass", "jailbreak", "dan"]
        danger_score = sum(1 for signal in danger_signals if signal in lower_text)
        
        if danger_score == 0:
            return 1.0 
        elif danger_score == 1:
            return 0.5 
        else:
            return 0.1 
            
    def evaluate(self, input_text):
        safety_score = self.mock_evaluator_llm(input_text)
        return safety_score >= self.safe_threshold

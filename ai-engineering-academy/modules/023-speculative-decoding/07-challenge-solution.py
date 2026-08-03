"""
AI ENGINEERING ACADEMY -- MODULE 023 ENGINEERING CHALLENGE SOLUTION
KV Cache Rollback Speculative Decoding Engine
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod23", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ToyLanguageModel   = _mod.ToyLanguageModel
RejectionSampler   = _mod.RejectionSampler
softmax            = _mod.softmax


class RollbackKVCache:
    """Simulated KV cache supporting length truncate/rollback."""

    def __init__(self):
        self.cached_len = 0

    def append(self, num_tokens=1):
        self.cached_len += num_tokens

    def rollback_to(self, target_len):
        self.cached_len = min(self.cached_len, target_len)


class KVCacheRollbackSpeculativeEngine:
    def __init__(self, target_model: ToyLanguageModel, draft_model: ToyLanguageModel, K=4, seed=42):
        self.target = target_model
        self.draft = draft_model
        self.K = K
        self.seed = seed
        self.target_kv_cache = RollbackKVCache()

    def generate(self, prompt_tokens, max_new_tokens=10):
        tokens = list(prompt_tokens)
        self.target_kv_cache.cached_len = len(prompt_tokens)

        target_passes = 0
        accepted_tokens_total = 0

        while len(tokens) - len(prompt_tokens) < max_new_tokens:
            # 1. Draft Phase
            draft_seq = list(tokens)
            q_probs_list = []
            draft_tokens_list = []

            for _ in range(self.K):
                q_logits = self.draft.forward(np.array([draft_seq]))[0, -1, :]
                q_prob = softmax(q_logits, temperature=1.0)
                next_tok = np.random.choice(self.draft.vocab_size, p=q_prob)

                q_probs_list.append(q_prob)
                draft_tokens_list.append(next_tok)
                draft_seq.append(next_tok)

            # 2. Target Parallel Pass
            eval_seq = np.array([draft_seq])
            target_logits = self.target.forward(eval_seq)[0]
            target_passes += 1

            start_pos = len(tokens) - 1
            p_probs = softmax(target_logits[start_pos : start_pos + self.K + 1], temperature=1.0)

            # 3. Rejection Sampling
            accepted_toks, num_acc = RejectionSampler.sample(
                p_probs, np.array(q_probs_list), np.array(draft_tokens_list), seed=self.seed
            )

            # Update token sequence & KV cache
            tokens.extend(accepted_toks)
            accepted_tokens_total += num_acc
            self.target_kv_cache.append(len(accepted_toks))

            # Truncate if token sequence exceeds desired length
            if len(tokens) - len(prompt_tokens) >= max_new_tokens:
                tokens = tokens[:len(prompt_tokens) + max_new_tokens]
                self.target_kv_cache.rollback_to(len(tokens))
                break

        return tokens, target_passes, accepted_tokens_total


def verify_kv_cache_rollback_speculative_engine():
    print("=" * 65)
    print("MODULE 023 CHALLENGE: KV CACHE ROLLBACK SPECULATIVE ENGINE")
    print("=" * 65)

    vocab_size = 50
    target_model = ToyLanguageModel(vocab_size=vocab_size, hidden_dim=32, seed=101)
    draft_model  = ToyLanguageModel(vocab_size=vocab_size, hidden_dim=8, seed=202)

    engine = KVCacheRollbackSpeculativeEngine(target_model, draft_model, K=4, seed=42)
    prompt = [2, 8, 14]

    generated_tokens, passes, accepted = engine.generate(prompt, max_new_tokens=10)

    print(f"Prompt Tokens:         {prompt}")
    print(f"Generated Tokens:      {generated_tokens[len(prompt):]}")
    print(f"Total Target Passes:   {passes} (vs 10 standard)")
    print(f"Accepted Draft Tokens: {accepted}")
    print(f"Final KV Cache Length: {engine.target_kv_cache.cached_len}")

    assert passes < 10
    assert engine.target_kv_cache.cached_len == len(generated_tokens)

    print("\nKV Cache Rollback Speculative Engine Verified Passed => [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_kv_cache_rollback_speculative_engine()

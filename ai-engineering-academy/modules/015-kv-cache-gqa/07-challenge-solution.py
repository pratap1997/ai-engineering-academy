"""
AI ENGINEERING ACADEMY -- MODULE 015 ENGINEERING CHALLENGE SOLUTION
Incremental Text Generator with GQA & KV Cache Verification
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod15", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

KVCache = _mod.KVCache
GroupedQueryAttention = _mod.GroupedQueryAttention


class IncrementalGQAGenerator:
    """
    Autoregressive generator using GQA and KV Cache.
    """

    def __init__(self, d_model=64, num_query_heads=8, num_kv_heads=2, seed=42):
        self.gqa = GroupedQueryAttention(d_model, num_query_heads, num_kv_heads, seed=seed)
        self.cache = KVCache()

    def generate(self, prompt, num_gen_steps=10):
        """
        prompt: (1, T_prompt, d_model)
        Returns: full_sequence (1, T_prompt + num_gen_steps, d_model), outputs_history
        """
        self.cache.reset()

        # Step 1: Prefill Phase
        prefill_out, _ = self.gqa.forward(prompt, kv_cache=self.cache)

        # Track history of sequence tokens
        generated_sequence = [prompt]
        curr_token = prefill_out[:, -1:, :]  # Use last token output as next input token

        # Step 2: Decoding Phase
        for _ in range(num_gen_steps):
            generated_sequence.append(curr_token)
            step_out, _ = self.gqa.forward(curr_token, kv_cache=self.cache)
            curr_token = step_out[:, -1:, :]

        full_seq = np.concatenate(generated_sequence, axis=1)
        return full_seq, self.cache.seq_len


def verify_kv_cache_generator():
    print("=" * 65)
    print("MODULE 015 CHALLENGE: INCREMENTAL GQA GENERATOR")
    print("=" * 65)

    np.random.seed(42)
    N, T_prompt, d_model = 1, 6, 64
    num_gen = 8

    generator = IncrementalGQAGenerator(d_model=d_model, num_query_heads=8, num_kv_heads=2, seed=42)
    prompt = np.random.randn(N, T_prompt, d_model)

    full_seq, final_cache_len = generator.generate(prompt, num_gen_steps=num_gen)

    print(f"Prompt Length:      {T_prompt} tokens")
    print(f"Generated Steps:    {num_gen} tokens")
    print(f"Final Sequence:     {full_seq.shape} (Expected: ({N}, {T_prompt + num_gen}, {d_model})) => [OK]")
    print(f"Final Cache Length: {final_cache_len} tokens (Expected: {T_prompt + num_gen}) => [OK]")

    assert final_cache_len == T_prompt + num_gen
    assert full_seq.shape == (N, T_prompt + num_gen, d_model)
    assert not np.isnan(full_seq).any()

    print("\nIncremental KV Cache GQA Generator Verified => [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_kv_cache_generator()

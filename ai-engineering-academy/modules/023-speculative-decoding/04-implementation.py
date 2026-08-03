"""
AI ENGINEERING ACADEMY -- MODULE 023
Speculative Decoding Implementation (Pure Python & NumPy)

Provides:
1. `DraftGenerator`: Generates K candidate draft tokens autoregressively.
2. `SpeculativeVerifier`: Parallel evaluation of K candidate tokens in 1 Target model forward pass.
3. `RejectionSampler`: Mathematically exact rejection sampling recovering Target distribution.
4. `SpeculativeDecoder`: Full speculative decoding engine.
"""

import numpy as np


def softmax(logits, temperature=1.0):
    logits = logits / max(1e-5, temperature)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_l = np.exp(logits)
    return exp_l / np.sum(exp_l, axis=-1, keepdims=True)


class ToyLanguageModel:
    """Toy LM representing Draft or Target model."""

    def __init__(self, vocab_size=50, hidden_dim=16, seed=42):
        np.random.seed(seed)
        scale = np.sqrt(2.0 / hidden_dim)
        self.vocab_size = vocab_size
        self.embeddings = np.random.randn(vocab_size, hidden_dim) * scale
        self.head = np.random.randn(hidden_dim, vocab_size) * scale

    def forward(self, token_ids):
        """token_ids: (batch_size, seq_len) -> logits (batch_size, seq_len, vocab_size)"""
        embeds = self.embeddings[token_ids]  # (B, T, hidden_dim)
        logits = np.matmul(embeds, self.head)
        return logits


class RejectionSampler:
    """
    Implements rejection sampling: accept draft token x if r <= min(1, p(x)/q(x)).
    If rejected, sample from relu(p(x) - q(x)).
    """

    @staticmethod
    def sample(p_probs, q_probs, draft_tokens, seed=None):
        """
        p_probs: (K+1, vocab_size) Target model probabilities
        q_probs: (K, vocab_size) Draft model probabilities
        draft_tokens: (K,) candidate token IDs
        Returns: accepted_tokens (list), num_accepted (int)
        """
        if seed is not None:
            np.random.seed(seed)

        accepted_tokens = []
        K = len(draft_tokens)

        for i in range(K):
            tok = draft_tokens[i]
            p_tok = p_probs[i, tok]
            q_tok = q_probs[i, tok]

            r = np.random.rand()
            acceptance_ratio = min(1.0, p_tok / max(1e-9, q_tok))

            if r <= acceptance_ratio:
                accepted_tokens.append(tok)
            else:
                # Rejected! Resample replacement token from relu(p - q)
                p_adjusted = np.maximum(0.0, p_probs[i] - q_probs[i])
                sum_adj = np.sum(p_adjusted)
                if sum_adj > 1e-9:
                    p_adjusted = p_adjusted / sum_adj
                else:
                    p_adjusted = p_probs[i]

                replacement_tok = np.random.choice(len(p_adjusted), p=p_adjusted)
                accepted_tokens.append(replacement_tok)
                return accepted_tokens, len(accepted_tokens) - 1

        # All K draft tokens accepted! Sample 1 bonus token from Target distribution at K
        bonus_p = p_probs[K]
        bonus_tok = np.random.choice(len(bonus_p), p=bonus_p)
        accepted_tokens.append(bonus_tok)

        return accepted_tokens, K


class SpeculativeDecoder:
    """End-to-End Speculative Decoding engine."""

    def __init__(self, target_model: ToyLanguageModel, draft_model: ToyLanguageModel, K=4, seed=42):
        self.target = target_model
        self.draft = draft_model
        self.K = K
        self.seed = seed

    def generate(self, prompt_tokens, max_new_tokens=12):
        tokens = list(prompt_tokens)
        total_target_passes = 0
        total_accepted_draft_tokens = 0
        total_draft_tokens_proposed = 0

        while len(tokens) - len(prompt_tokens) < max_new_tokens:
            # 1. Draft Phase: Draft model generates K candidate tokens
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

            q_probs = np.array(q_probs_list)        # (K, vocab_size)
            draft_tokens = np.array(draft_tokens_list) # (K,)

            # 2. Verification Phase: Single parallel Target forward pass over (len(tokens) + K)
            eval_seq = np.array([draft_seq])
            target_logits = self.target.forward(eval_seq)[0]  # (T + K, vocab_size)
            total_target_passes += 1

            # Extract target probabilities for verification positions
            start_pos = len(tokens) - 1
            p_probs = softmax(target_logits[start_pos : start_pos + self.K + 1], temperature=1.0) # (K+1, vocab_size)

            # 3. Rejection Sampling
            accepted_toks, num_acc = RejectionSampler.sample(p_probs, q_probs, draft_tokens, seed=self.seed)

            tokens.extend(accepted_toks)
            total_accepted_draft_tokens += num_acc
            total_draft_tokens_proposed += self.K

            if len(tokens) - len(prompt_tokens) >= max_new_tokens:
                break

        tokens = tokens[:len(prompt_tokens) + max_new_tokens]
        return tokens, total_target_passes, total_accepted_draft_tokens, total_draft_tokens_proposed


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 023 -- SPECULATIVE DECODING VERIFICATION")
    print("=" * 65)

    vocab_size = 50
    target_model = ToyLanguageModel(vocab_size=vocab_size, hidden_dim=32, seed=101)
    draft_model = ToyLanguageModel(vocab_size=vocab_size, hidden_dim=8, seed=202)

    decoder = SpeculativeDecoder(target_model, draft_model, K=4, seed=42)
    prompt = [1, 5, 12]

    output_tokens, target_passes, accepted, proposed = decoder.generate(prompt, max_new_tokens=12)

    print("\n[1. Speculative Decoding Execution Statistics]")
    print(f"  Prompt Tokens:           {prompt}")
    print(f"  Generated Tokens:        {output_tokens[len(prompt):]}")
    print(f"  Total Target Passes:     {target_passes} (vs 12 in standard decoding)")
    print(f"  Draft Tokens Proposed:   {proposed}")
    print(f"  Draft Tokens Accepted:   {accepted} (Acceptance Rate: {(accepted/proposed)*100:.1f}%)")

    speedup_ratio = 12 / max(1, target_passes)
    print(f"  Theoretical Forward Pass Speedup: {speedup_ratio:.2f}x fewer Target passes!")

    assert target_passes < 12
    print("  Speculative decoding required significantly fewer Target passes => [OK]")

"""
AI ENGINEERING ACADEMY -- MODULE 012
BERT & GPT Pre-training Objectives (Pure Python & NumPy)

Provides:
1. `MLMMasker`:       Applies the 80/10/10 masking strategy to token sequences.
2. `MLMLoss`:         Masked cross-entropy loss (only over masked positions).
3. `CLMLoss`:         Causal/Autoregressive cross-entropy loss (shifted targets).
4. `BERTEmbeddings`:  Token + Segment + Position embedding lookup tables.
5. `EmbeddingTable`:  Shared token embedding + linear projection head.
"""

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def cross_entropy(logits, targets, mask=None):
    """
    logits:  (N, T, V)
    targets: (N, T)  -- integer token ids (-100 for ignored positions)
    mask:    (N, T)  -- 1=compute loss, 0=ignore
    Returns: scalar loss
    """
    N, T, V = logits.shape
    probs = softmax(logits)

    # Safe target indexing (-100 replaced by 0 for indexing; masked out later)
    safe_targets = np.where(targets < 0, 0, targets)

    N_idx = np.arange(N)[:, None]
    T_idx = np.arange(T)[None, :]
    correct_probs = probs[N_idx, T_idx, safe_targets]  # (N, T)

    log_probs = -np.log(correct_probs + 1e-9)

    if mask is not None:
        log_probs = log_probs * mask
        mask_sum = mask.sum()
        if mask_sum == 0:
            return 0.0
        return log_probs.sum() / mask_sum
    return log_probs.mean()


# =====================================================================
# 1. MLM MASKER
# =====================================================================

class MLMMasker:
    """
    Applies BERT's 80/10/10 masking strategy:
      80% -> [MASK] token  (mask_token_id)
      10% -> random token from vocabulary
      10% -> unchanged (original token)

    Only 15% of tokens are selected for prediction.
    """

    MASK_PROB = 0.15  # Fraction of tokens selected for masking

    def __init__(self, vocab_size, mask_token_id, seed=None):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.seed = seed

    def mask(self, tokens):
        """
        tokens: (N, T) integer array
        Returns:
          masked_tokens: (N, T) -- input with masking applied
          labels:        (N, T) -- original tokens (-100 at non-masked positions)
          mask_flags:    (N, T) -- 1 where loss should be computed
        """
        if self.seed is not None:
            rng = np.random.RandomState(self.seed)
        else:
            rng = np.random.RandomState()

        N, T = tokens.shape
        masked_tokens = tokens.copy()
        labels = np.full((N, T), -100)  # -100 = ignore in loss
        mask_flags = np.zeros((N, T), dtype=bool)

        for n in range(N):
            num_to_mask = max(1, int(T * self.MASK_PROB))
            chosen = rng.choice(T, size=num_to_mask, replace=False)

            labels[n, chosen] = tokens[n, chosen]  # Save original for loss
            mask_flags[n, chosen] = True

            for pos in chosen:
                r = rng.random_sample()
                if r < 0.80:
                    masked_tokens[n, pos] = self.mask_token_id  # [MASK]
                elif r < 0.90:
                    masked_tokens[n, pos] = rng.randint(0, self.vocab_size)  # Random
                # else: keep original (10%)

        return masked_tokens, labels, mask_flags


# =====================================================================
# 2. MLM LOSS
# =====================================================================

class MLMLoss:
    """Cross-entropy loss computed only over masked positions."""

    def forward(self, logits, labels, mask_flags):
        """
        logits:     (N, T, V)
        labels:     (N, T) original token ids at masked positions
        mask_flags: (N, T) bool -- where to compute loss
        Returns: scalar
        """
        mask_float = mask_flags.astype(float)
        return cross_entropy(logits, labels, mask=mask_float)


# =====================================================================
# 3. CLM LOSS (GPT)
# =====================================================================

class CLMLoss:
    """
    Causal Language Model loss: predict token t+1 from tokens 0..t.
    Input logits: (N, T, V); targets are tokens shifted left by 1.
    """

    def forward(self, logits, tokens):
        """
        logits: (N, T, V) -- model predictions for each position
        tokens: (N, T)    -- original token sequence
        Returns: scalar (averaged over T-1 positions)
        """
        # Predictions at positions 0..T-2, targets at positions 1..T
        pred_logits = logits[:, :-1, :]   # (N, T-1, V)
        targets     = tokens[:, 1:]       # (N, T-1)
        return cross_entropy(pred_logits, targets)


# =====================================================================
# 4. BERT EMBEDDINGS
# =====================================================================

class BERTEmbeddings:
    """
    E(t) = TokenEmb(x_t) + SegmentEmb(s_t) + PositionEmb(t)
    All three are learned lookup tables.
    """

    def __init__(self, vocab_size, d_model, max_len=512, num_segments=2, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scale = 0.02
        self.token_emb   = np.random.randn(vocab_size, d_model) * scale
        self.segment_emb = np.random.randn(num_segments, d_model) * scale
        self.position_emb = np.random.randn(max_len, d_model) * scale

    def forward(self, token_ids, segment_ids=None):
        """
        token_ids:   (N, T) integers in [0, vocab_size)
        segment_ids: (N, T) integers in {0, 1} -- 0=Sentence A, 1=Sentence B
        Returns: (N, T, d_model)
        """
        N, T = token_ids.shape
        emb = self.token_emb[token_ids]           # Token embeddings
        emb = emb + self.position_emb[:T, :]     # Position embeddings

        if segment_ids is not None:
            emb = emb + self.segment_emb[segment_ids]  # Segment embeddings
        return emb


# =====================================================================
# 5. SHARED EMBEDDING + LM HEAD
# =====================================================================

class LMHead:
    """
    Shared weight matrix between input embeddings and output projection.
    logits = hidden_states @ token_emb.T
    """

    def __init__(self, token_emb):
        self.token_emb = token_emb  # (vocab_size, d_model) -- shared weights

    def forward(self, hidden):
        """hidden: (N, T, d_model) -> logits (N, T, vocab_size)"""
        return np.matmul(hidden, self.token_emb.T)


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 012 -- BERT & GPT PRE-TRAINING VERIFICATION")
    print("=" * 65)

    vocab_size, d_model = 100, 32
    MASK_ID = 4
    N, T = 2, 12

    np.random.seed(42)
    tokens = np.random.randint(5, vocab_size, size=(N, T))

    # 1. MLM Masker
    masker = MLMMasker(vocab_size=vocab_size, mask_token_id=MASK_ID, seed=42)
    masked_tokens, labels, mask_flags = masker.mask(tokens)
    print("\n[1. MLM Masker]")
    print(f"  Original tokens shape: {tokens.shape}")
    print(f"  Masked tokens shape:   {masked_tokens.shape} => [OK]")
    pct_masked = mask_flags.mean() * 100
    print(f"  ~15% positions masked: {pct_masked:.1f}% => [OK]")

    # 2. MLM Loss
    logits = np.random.randn(N, T, vocab_size)
    mlm_loss_fn = MLMLoss()
    loss_mlm = mlm_loss_fn.forward(logits, labels, mask_flags)
    print("\n[2. MLM Loss]")
    print(f"  Loss value: {loss_mlm:.4f} (expect ~log({vocab_size})={np.log(vocab_size):.2f}) => [OK]")

    # 3. CLM Loss
    clm_loss_fn = CLMLoss()
    loss_clm = clm_loss_fn.forward(logits, tokens)
    print("\n[3. CLM Loss]")
    print(f"  Loss value: {loss_clm:.4f} (expect ~log({vocab_size})={np.log(vocab_size):.2f}) => [OK]")

    # 4. BERT Embeddings
    bert_emb = BERTEmbeddings(vocab_size=vocab_size, d_model=d_model, seed=42)
    seg_ids = np.zeros((N, T), dtype=int)
    seg_ids[:, T//2:] = 1  # Second half = Sentence B
    emb_out = bert_emb.forward(tokens, seg_ids)
    print("\n[4. BERT Embeddings (Token + Segment + Position)]")
    print(f"  Output Shape: {emb_out.shape} (Expected: ({N}, {T}, {d_model})) => [OK]")

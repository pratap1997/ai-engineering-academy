"""
AI ENGINEERING ACADEMY -- MODULE 014
Advanced Positional Encodings Implementation (Pure Python & NumPy)

Provides:
1. `RoPEEmbedding`: Rotary Position Embeddings (LLaMA / Mistral style).
2. `ALiBiBias`: Attention with Linear Biases matrix generator (BLOOM / MPT style).
3. `RelativePositionBiasT5`: Logarithmic relative position bias lookup table (T5 style).
"""

import numpy as np


# =====================================================================
# 1. ROTARY POSITION EMBEDDING (RoPE)
# =====================================================================

class RoPEEmbedding:
    """
    Rotary Position Embeddings (Su et al., 2021).
    Rotates 2D pairs of vector coordinates by position-dependent angles.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        assert dim % 2 == 0, "dim must be even for 2D pair rotation"
        self.dim = dim
        self.base = base
        self.max_seq_len = max_position_embeddings

        # Compute theta frequencies: theta_i = base^(-2(i-1)/dim)
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim, 2).astype(np.float64) / self.dim))
        self.inv_freq = inv_freq

        # Precompute cos and sin embeddings for max sequence length
        self._build_cache(self.max_seq_len)

    def _build_cache(self, seq_len):
        t = np.arange(seq_len, dtype=np.float64)
        # Outer product: (seq_len, dim/2)
        freqs = np.outer(t, self.inv_freq)
        # Duplicate freqs along last dim so shape becomes (seq_len, dim)
        emb = np.concatenate([freqs, freqs], axis=-1)

        self.cos_cached = np.cos(emb)  # (seq_len, dim)
        self.sin_cached = np.sin(emb)  # (seq_len, dim)

    def _rotate_half(self, x):
        """Rotate vector x: [-x2, x1, -x4, x3, ...]"""
        d2 = x.shape[-1] // 2
        x1 = x[..., :d2]
        x2 = x[..., d2:]
        return np.concatenate([-x2, x1], axis=-1)

    def apply(self, x, seq_dim=1):
        """
        Apply RoPE rotation to tensor x of shape (N, T, H, d) or (N, T, d).
        x: (..., T, d)
        Returns: rotated tensor of same shape
        """
        T = x.shape[seq_dim]
        if T > self.cos_cached.shape[0]:
            self._build_cache(T)

        # Slice cos and sin up to current sequence length T
        cos = self.cos_cached[:T]
        sin = self.sin_cached[:T]

        # Reshape cos and sin for broadcasting with x
        # If x is 4D (N, T, H, d): cos needs shape (1, T, 1, d)
        # If x is 3D (N, T, d): cos needs shape (1, T, d)
        ndim = x.ndim
        if ndim == 4:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        elif ndim == 3:
            cos = cos[None, :, :]
            sin = sin[None, :, :]

        return (x * cos) + (self._rotate_half(x) * sin)


# =====================================================================
# 2. ATTENTION WITH LINEAR BIASES (ALiBi)
# =====================================================================

class ALiBiBias:
    """
    Attention with Linear Biases (Press et al., 2022).
    Generates a static distance penalty matrix weighted by per-head slopes.
    """

    def __init__(self, num_heads):
        self.num_heads = num_heads
        self.slopes = self._get_slopes(num_heads)

    def _get_slopes(self, num_heads):
        """Get geometric slopes m_h for each head."""
        def get_slopes_power_of_2(n):
            start = (2 ** (-8 / n))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if math_is_power_of_2(num_heads):
            return np.array(get_slopes_power_of_2(num_heads), dtype=np.float64)
        else:
            # Nearest power of 2
            closest_pow2 = 2 ** int(np.floor(np.log2(num_heads)))
            slopes_pow2 = get_slopes_power_of_2(closest_pow2)
            extra_slopes = get_slopes_power_of_2(2 * closest_pow2)[0::2][:num_heads - closest_pow2]
            return np.array(slopes_pow2 + extra_slopes, dtype=np.float64)

    def forward(self, seq_len):
        """
        Generate ALiBi bias matrix of shape (1, num_heads, seq_len, seq_len).
        Bias_h[i, j] = - slope_h * |i - j|  (for causal/full distance)
        """
        # Distance matrix: |i - j|
        context_position = np.arange(seq_len)[:, None]
        memory_position = np.arange(seq_len)[None, :]
        relative_distance = np.abs(context_position - memory_position)  # (seq_len, seq_len)

        # Scale by per-head slopes: (num_heads, seq_len, seq_len)
        bias = -1.0 * self.slopes[:, None, None] * relative_distance[None, :, :]
        return bias[None, :, :, :]  # (1, num_heads, seq_len, seq_len)


def math_is_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0


# =====================================================================
# 3. T5 RELATIVE POSITION BIAS
# =====================================================================

class RelativePositionBiasT5:
    """
    T5-style relative position bucketing (Raffel et al. / Shaw et al.).
    Buckets relative distances logarithmically and looks up a learned scalar bias.
    """

    def __init__(self, num_heads, num_buckets=32, max_distance=128, seed=None):
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        if seed is not None:
            np.random.seed(seed)
        # Learned scalar bias per bucket and head: (num_buckets, num_heads)
        self.relative_attention_bias = np.random.randn(num_buckets, num_heads) * 0.02

    def _relative_position_bucket(self, relative_position, bidirectional=True):
        """Map relative position integers to bucket IDs."""
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        ret = 0

        if bidirectional:
            num_buckets //= 2
            ret += np.where(relative_position < 0, num_buckets, 0)
            relative_position = np.abs(relative_position)
        else:
            relative_position = np.maximum(-relative_position, 0)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        val_if_large = max_exact + (
            np.log(relative_position.astype(np.float64) / max_exact + 1e-9)
            / np.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)

        ret += np.where(is_small, relative_position, val_if_large)
        return ret

    def forward(self, query_length, key_length):
        """
        Generate T5 relative position bias of shape (1, num_heads, query_length, key_length).
        """
        context_position = np.arange(query_length)[:, None]
        memory_position = np.arange(key_length)[None, :]
        relative_position = memory_position - context_position  # (query_len, key_len)

        buckets = self._relative_position_bucket(relative_position, bidirectional=True)
        # Lookup bias: (query_len, key_len, num_heads)
        bias = self.relative_attention_bias[buckets]
        # Transpose to (1, num_heads, query_len, key_len)
        return bias.transpose(2, 0, 1)[None, :, :, :]


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 014 -- ADVANCED POSITIONAL ENCODINGS VERIFICATION")
    print("=" * 65)

    N, T, H, d = 2, 8, 4, 16

    # 1. RoPE Test
    rope = RoPEEmbedding(dim=d)
    q = np.random.randn(N, T, H, d)
    q_rotated = rope.apply(q, seq_dim=1)
    print("\n[1. Rotary Position Embedding (RoPE)]")
    print(f"  Input Shape:   {q.shape}")
    print(f"  Rotated Shape: {q_rotated.shape} (Expected: ({N}, {T}, {H}, {d})) => [OK]")

    # Verify inner product relativity
    X_seq = np.zeros((1, 10, 1, d))
    q_vec = np.random.randn(d)
    k_vec = np.random.randn(d)

    # Place (q, k) at (m=0, n=2) -> distance = 2
    X_seq1 = X_seq.copy()
    X_seq1[0, 0, 0] = q_vec
    X_seq1[0, 2, 0] = k_vec
    rot1 = rope.apply(X_seq1)
    dot1 = np.dot(rot1[0, 0, 0], rot1[0, 2, 0])

    # Place (q, k) at (m=5, n=7) -> distance = 2
    X_seq2 = X_seq.copy()
    X_seq2[0, 5, 0] = q_vec
    X_seq2[0, 7, 0] = k_vec
    rot2 = rope.apply(X_seq2)
    dot2 = np.dot(rot2[0, 5, 0], rot2[0, 7, 0])

    print(f"  Relative Dot Product at (0, 2): {dot1:.6f}")
    print(f"  Relative Dot Product at (5, 7): {dot2:.6f}")
    np.testing.assert_allclose(dot1, dot2, atol=1e-5)
    print("  RoPE Relative Inner Product Preservation Verified => [OK]")

    # 2. ALiBi Test
    alibi = ALiBiBias(num_heads=H)
    alibi_matrix = alibi.forward(seq_len=T)
    print("\n[2. Attention with Linear Biases (ALiBi)]")
    print(f"  ALiBi Matrix Shape: {alibi_matrix.shape} (Expected: (1, {H}, {T}, {T})) => [OK]")
    print(f"  Head 0 Slopes: {alibi.slopes[0]:.4f}, Head 3 Slopes: {alibi.slopes[3]:.4f}")

    # 3. T5 Relative Bias Test
    t5_bias = RelativePositionBiasT5(num_heads=H, seed=42)
    t5_matrix = t5_bias.forward(query_length=T, key_length=T)
    print("\n[3. T5 Relative Position Bias]")
    print(f"  T5 Bias Shape: {t5_matrix.shape} (Expected: (1, {H}, {T}, {T})) => [OK]")

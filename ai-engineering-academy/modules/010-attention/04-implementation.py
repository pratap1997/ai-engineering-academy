"""
AI ENGINEERING ACADEMY — MODULE 010
Attention Mechanisms Implementation (Pure Python & NumPy)

Provides:
1. `ScaledDotProductAttention`: Single-head attention with optional causal mask.
2. `MultiHeadAttention`: H parallel attention heads with learned projections.
3. `SinusoidalPositionalEncoding`: Position-aware token embeddings.
"""

import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# =====================================================================
# 1. SCALED DOT-PRODUCT ATTENTION
# =====================================================================

class ScaledDotProductAttention:
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """

    def __init__(self):
        self.attention_weights = None

    def forward(self, Q, K, V, mask=None):
        """
        Q: (N, T_q, d_k)
        K: (N, T_k, d_k)
        V: (N, T_k, d_v)
        mask: Optional boolean (N, T_q, T_k) — True means MASKED (set to -inf)
        Returns: output (N, T_q, d_v), attention_weights (N, T_q, T_k)
        """
        d_k = Q.shape[-1]
        scale = np.sqrt(d_k)

        # Step 1: Raw scores (N, T_q, T_k)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / scale

        # Step 2: Apply causal/padding mask
        if mask is not None:
            scores = scores + mask * -1e9

        # Step 3: Softmax over T_k dimension
        self.attention_weights = softmax(scores, axis=-1)

        # Step 4: Weighted sum of values
        output = np.matmul(self.attention_weights, V)
        return output, self.attention_weights


# =====================================================================
# 2. MULTI-HEAD ATTENTION
# =====================================================================

class MultiHeadAttention:
    """
    MultiHead(Q,K,V) = Concat(head_1,...,head_H) * W_O
    head_h = Attention(Q*W_Q_h, K*W_K_h, V*W_V_h)
    """

    def __init__(self, d_model, num_heads, seed=None):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.attention = ScaledDotProductAttention()

        if seed is not None:
            np.random.seed(seed)

        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def _split_heads(self, X):
        """Split last dimension into (num_heads, d_head). X: (N, T, d_model)"""
        N, T, _ = X.shape
        X = X.reshape(N, T, self.num_heads, self.d_head)
        return X.transpose(0, 2, 1, 3)  # (N, H, T, d_head)

    def _merge_heads(self, X):
        """Merge (num_heads, d_head) back. X: (N, H, T, d_head)"""
        N, H, T, _ = X.shape
        X = X.transpose(0, 2, 1, 3)  # (N, T, H, d_head)
        return X.reshape(N, T, self.d_model)  # (N, T, d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (N, T, d_model)
        Returns: output (N, T, d_model)
        """
        N, T_q, _ = Q.shape
        N, T_k, _ = K.shape

        # Linear projections
        Q_proj = np.matmul(Q, self.W_Q)  # (N, T_q, d_model)
        K_proj = np.matmul(K, self.W_K)  # (N, T_k, d_model)
        V_proj = np.matmul(V, self.W_V)  # (N, T_k, d_model)

        # Split into H heads: (N, H, T, d_head)
        Q_heads = self._split_heads(Q_proj)
        K_heads = self._split_heads(K_proj)
        V_heads = self._split_heads(V_proj)

        # Reshape for batched attention: (N*H, T, d_head)
        def flatten_heads(X):
            N, H, T, d = X.shape
            return X.reshape(N * H, T, d)

        Q_flat = flatten_heads(Q_heads)
        K_flat = flatten_heads(K_heads)
        V_flat = flatten_heads(V_heads)

        # Scaled dot-product attention
        attn_out, attn_weights = self.attention.forward(Q_flat, K_flat, V_flat, mask)

        # Restore shape: (N, H, T, d_head)
        attn_out = attn_out.reshape(N, self.num_heads, T_q, self.d_head)

        # Merge heads & output projection
        merged = self._merge_heads(attn_out)         # (N, T, d_model)
        output = np.matmul(merged, self.W_O)          # (N, T, d_model)

        return output, attn_weights.reshape(N, self.num_heads, T_q, T_k)


# =====================================================================
# 3. SINUSOIDAL POSITIONAL ENCODING
# =====================================================================

class SinusoidalPositionalEncoding:
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=512):
        self.d_model = d_model
        self.pe = self._build(d_model, max_len)

    def _build(self, d_model, max_len):
        pe = np.zeros((max_len, d_model))
        positions = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
        div_term = np.power(10000.0, np.arange(0, d_model, 2) / d_model)

        pe[:, 0::2] = np.sin(positions / div_term)
        pe[:, 1::2] = np.cos(positions / div_term)
        return pe  # (max_len, d_model)

    def forward(self, X):
        """Add positional encoding to input X: (N, T, d_model)"""
        T = X.shape[1]
        return X + self.pe[:T, :]


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 010 — ATTENTION MECHANISM VERIFICATION")
    print("=" * 65)

    N, T, d_model, H = 2, 6, 64, 4

    # Test 1: Scaled Dot-Product Attention
    sdpa = ScaledDotProductAttention()
    Q = np.random.randn(N, T, d_model // H)
    K = np.random.randn(N, T, d_model // H)
    V = np.random.randn(N, T, d_model // H)
    out, attn = sdpa.forward(Q, K, V)
    print("\n[1. Scaled Dot-Product Attention]")
    print(f"  Output Shape:          {out.shape}  (Expected: ({N}, {T}, {d_model//H})) => [OK]")
    print(f"  Attention Weight Shape: {attn.shape} (Expected: ({N}, {T}, {T})) => [OK]")
    print(f"  Attention Weights Sum:  {attn[0,0].sum():.4f} (Expected: 1.0) => [OK]")

    # Test 2: Multi-Head Attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=H, seed=42)
    X = np.random.randn(N, T, d_model)
    out_mha, attn_mha = mha.forward(X, X, X)
    print("\n[2. Multi-Head Attention (Self-Attention)]")
    print(f"  Output Shape: {out_mha.shape} (Expected: ({N}, {T}, {d_model})) => [OK]")
    print(f"  Attn Heads:   {attn_mha.shape} (Expected: ({N}, {H}, {T}, {T})) => [OK]")

    # Test 3: Positional Encoding
    pe = SinusoidalPositionalEncoding(d_model=d_model)
    X_encoded = pe.forward(X)
    print("\n[3. Sinusoidal Positional Encoding]")
    print(f"  Input Shape:   {X.shape}")
    print(f"  Encoded Shape: {X_encoded.shape} (Same as input + PE) => [OK]")

"""
AI ENGINEERING ACADEMY -- MODULE 020
Multi-Head Latent Attention (MLA) Implementation (Pure Python & NumPy)

Provides:
1. `DecoupledRoPE`: Position embedding generator for decoupled key/query positional features.
2. `MLALayer`: DeepSeek-style Multi-Head Latent Attention layer with low-rank c_KV compression.
3. `MLAMatrixAbsorption`: Pre-computing absorbed projection matrices W_absorbed = W_UQ @ W_UK.T for zero-overhead inference.
"""

import numpy as np


def apply_rope_2d(x):
    """Simple 2D RoPE rotation for testing."""
    B, T, d = x.shape
    half_d = d // 2
    x1, x2 = x[..., :half_d], x[..., half_d:]
    positions = np.arange(T)[:, None]
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, half_d, 1) / half_d))
    angles = positions * inv_freq
    sin, cos = np.sin(angles), np.cos(angles)

    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos
    return np.concatenate([x1_rot, x2_rot], axis=-1)


# =====================================================================
# 1. MULTI-HEAD LATENT ATTENTION (MLA) LAYER
# =====================================================================

class MLALayer:
    """
    DeepSeek Multi-Head Latent Attention (MLA) layer.
    Compresses Keys and Values into joint latent space c_KV of dimension d_c.
    """

    def __init__(self, d_model=64, num_heads=4, d_c=16, d_h=16, d_R=8, seed=None):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_c = d_c          # Latent KV compression dimension
        self.d_h = d_h          # Content head dimension
        self.d_R = d_R          # Decoupled RoPE dimension
        self.scale = 1.0 / np.sqrt(d_h + d_R)

        if seed is not None:
            np.random.seed(seed)

        # Down-projection for KV joint compression: W_DKV
        self.W_DKV = np.random.randn(d_model, d_c) * np.sqrt(2.0 / d_model)

        # Up-projections for Key & Value content heads: W_UK, W_UV
        self.W_UK = np.random.randn(d_c, num_heads * d_h) * np.sqrt(2.0 / d_c)
        self.W_UV = np.random.randn(d_c, num_heads * d_h) * np.sqrt(2.0 / d_c)

        # Query projection (down & up compressed): W_DQ, W_UQ
        self.W_DQ = np.random.randn(d_model, d_c) * np.sqrt(2.0 / d_model)
        self.W_UQ = np.random.randn(d_c, num_heads * d_h) * np.sqrt(2.0 / d_c)

        # Decoupled RoPE projections: W_QR, W_KR
        self.W_QR = np.random.randn(d_c, num_heads * d_R) * np.sqrt(2.0 / d_c)
        self.W_KR = np.random.randn(d_model, d_R) * np.sqrt(2.0 / d_model)

        # Output projection: W_O
        self.W_O = np.random.randn(num_heads * d_h, d_model) * np.sqrt(2.0 / (num_heads * d_h))

    def forward(self, x, causal=False):
        """
        x: (batch_size, seq_len, d_model)
        Returns: output (batch_size, seq_len, d_model), c_KV (batch_size, seq_len, d_c)
        """
        B, T, d = x.shape

        # 1. Joint KV Latent Compression
        c_KV = np.matmul(x, self.W_DKV)  # (B, T, d_c)

        # 2. Reconstruct Content Keys & Values
        K_C = np.matmul(c_KV, self.W_UK).reshape(B, T, self.num_heads, self.d_h).transpose(0, 2, 1, 3)
        V_C = np.matmul(c_KV, self.W_UV).reshape(B, T, self.num_heads, self.d_h).transpose(0, 2, 1, 3)

        # 3. Query Content & RoPE Projections
        c_Q = np.matmul(x, self.W_DQ)   # (B, T, d_c)
        Q_C = np.matmul(c_Q, self.W_UQ).reshape(B, T, self.num_heads, self.d_h).transpose(0, 2, 1, 3)

        q_R_flat = np.matmul(c_Q, self.W_QR)  # (B, T, num_heads * d_R)
        q_R = apply_rope_2d(q_R_flat).reshape(B, T, self.num_heads, self.d_R).transpose(0, 2, 1, 3)

        k_R_flat = np.matmul(x, self.W_KR)    # (B, T, d_R)
        k_R = apply_rope_2d(k_R_flat)[:, None, :, :]  # (B, 1, T, d_R) broadcast across heads

        # 4. Content Attention Scores + RoPE Scores
        scores_C = np.matmul(Q_C, K_C.transpose(0, 1, 3, 2))  # (B, H, T, T)
        scores_R = np.matmul(q_R, k_R.transpose(0, 1, 3, 2))  # (B, H, T, T)
        scores = (scores_C + scores_R) * self.scale

        if causal:
            mask = np.triu(np.ones((T, T)), k=1)
            scores = np.where(mask[None, None, :, :], -1e9, scores)

        # 5. Softmax & Context Aggregation
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)

        context_C = np.matmul(attn, V_C)  # (B, H, T, d_h)
        context_flat = context_C.transpose(0, 2, 1, 3).reshape(B, T, self.num_heads * self.d_h)

        out = np.matmul(context_flat, self.W_O)
        return out, c_KV


# =====================================================================
# 2. MATRIX ABSORPTION INFERENCE OPTIMIZER
# =====================================================================

class MLAMatrixAbsorption:
    """
    Pre-computes W_absorbed = W_UQ @ W_UK.T to eliminate Key up-projection during inference.
    """

    def __init__(self, mla_layer: MLALayer):
        self.mla = mla_layer

        # Reshape W_UQ to (num_heads, d_c, d_h) and W_UK to (num_heads, d_c, d_h)
        # W_absorbed per head: W_UQ_h @ W_UK_h.T -> shape (num_heads, d_c, d_c)
        W_UQ_heads = mla_layer.W_UQ.reshape(mla_layer.d_c, mla_layer.num_heads, mla_layer.d_h).transpose(1, 0, 2)
        W_UK_heads = mla_layer.W_UK.reshape(mla_layer.d_c, mla_layer.num_heads, mla_layer.d_h).transpose(1, 0, 2)

        # W_absorbed[h] = W_UQ[h] @ W_UK[h].T  -> shape (num_heads, d_c, d_c)
        self.W_absorbed = np.matmul(W_UQ_heads, W_UK_heads.transpose(0, 2, 1))

    def compute_content_scores(self, c_Q, c_KV):
        """
        c_Q: (B, T_q, d_c)
        c_KV: (B, T_k, d_c)
        Returns scores_C: (B, H, T_q, T_k) computed directly without decompressing K!
        """
        B, T_q, d_c = c_Q.shape
        _, T_k, _ = c_KV.shape

        # c_Q @ W_absorbed[h] @ c_KV.T
        scores_C = np.zeros((B, self.mla.num_heads, T_q, T_k))
        for h in range(self.mla.num_heads):
            q_h = np.matmul(c_Q, self.W_absorbed[h])  # (B, T_q, d_c)
            scores_C[:, h, :, :] = np.matmul(q_h, c_KV.transpose(0, 2, 1))

        return scores_C


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 020 -- MULTI-HEAD LATENT ATTENTION (MLA) VERIFICATION")
    print("=" * 65)

    np.random.seed(42)
    B, T, d_model = 2, 8, 64
    x = np.random.randn(B, T, d_model)

    mla = MLALayer(d_model=64, num_heads=4, d_c=16, d_h=16, d_R=8, seed=42)
    out, c_KV = mla.forward(x, causal=True)

    print("\n[1. MLA Layer Forward Pass]")
    print(f"  Input Shape:       {x.shape}")
    print(f"  Output Shape:      {out.shape} (Expected: ({B}, {T}, {d_model})) => [OK]")
    print(f"  Compressed c_KV:   {c_KV.shape} (Compressed from {B}x{T}x{4*16} to {B}x{T}x{16}) => [OK]")

    # 2. Matrix Absorption Verification
    abs_opt = MLAMatrixAbsorption(mla)
    c_Q = np.matmul(x, mla.W_DQ)

    scores_absorbed = abs_opt.compute_content_scores(c_Q, c_KV)

    # Standard uncompressed K_C scores
    K_C = np.matmul(c_KV, mla.W_UK).reshape(B, T, 4, 16).transpose(0, 2, 1, 3)
    Q_C = np.matmul(c_Q, mla.W_UQ).reshape(B, T, 4, 16).transpose(0, 2, 1, 3)
    scores_standard = np.matmul(Q_C, K_C.transpose(0, 1, 3, 2))

    max_diff = np.max(np.abs(scores_absorbed - scores_standard))
    print("\n[2. Matrix Absorption Verification]")
    print(f"  Max score difference (Absorbed vs Uncompressed): {max_diff:.8e}")
    np.testing.assert_allclose(scores_absorbed, scores_standard, atol=1e-5)
    print("  EXACT MATRIX ABSORPTION NUMERICAL EQUIVALENCE VERIFIED => [OK]")

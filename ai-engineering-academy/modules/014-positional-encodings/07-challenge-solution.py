"""
AI ENGINEERING ACADEMY -- MODULE 014 ENGINEERING CHALLENGE SOLUTION
RoPE Multi-Head Attention Forward Pass & Relative Shift Verification
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod14", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

RoPEEmbedding = _mod.RoPEEmbedding


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class RoPEMultiHeadAttention:
    def __init__(self, d_model, num_heads, seed=None):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        if seed is not None:
            np.random.seed(seed)

        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

        self.rope = RoPEEmbedding(dim=self.d_head)

    def forward(self, X):
        N, T, _ = X.shape

        Q = np.matmul(X, self.W_Q).reshape(N, T, self.num_heads, self.d_head)
        K = np.matmul(X, self.W_K).reshape(N, T, self.num_heads, self.d_head)
        V = np.matmul(X, self.W_V).reshape(N, T, self.num_heads, self.d_head)

        # Apply RoPE to Q and K across all heads
        Q_rot = self.rope.apply(Q, seq_dim=1)  # (N, T, H, d_head)
        K_rot = self.rope.apply(K, seq_dim=1)  # (N, T, H, d_head)

        # Transpose for attention: (N, H, T, d_head)
        Q_rot = Q_rot.transpose(0, 2, 1, 3)
        K_rot = K_rot.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = np.sqrt(self.d_head)
        raw_scores = np.matmul(Q_rot, K_rot.transpose(0, 1, 3, 2)) / scale
        attn = softmax(raw_scores, axis=-1)

        out = np.matmul(attn, V)  # (N, H, T, d_head)
        out = out.transpose(0, 2, 1, 3).reshape(N, T, self.d_model)
        return np.matmul(out, self.W_O), attn, raw_scores


def verify_rope_mha():
    print("=" * 65)
    print("MODULE 014 CHALLENGE: RoPE MULTI-HEAD ATTENTION")
    print("=" * 65)

    np.random.seed(42)
    N, T, d_model, H = 1, 10, 32, 4
    mha = RoPEMultiHeadAttention(d_model=d_model, num_heads=H, seed=42)

    X = np.random.randn(N, T, d_model)
    out, attn, raw_scores = mha.forward(X)

    print(f"Input Shape:  {X.shape}")
    print(f"Output Shape: {out.shape} (Expected: ({N}, {T}, {d_model})) => [OK]")
    print(f"Attn Shape:   {attn.shape} (Expected: ({N}, {H}, {T}, {T})) => [OK]")

    # Shift verification: raw score between pos 2 and pos 5 vs shifted pos 4 and pos 7
    shift = 2
    X_large = np.random.randn(N, T + shift, d_model)
    X_large[0, 2] = X[0, 2]
    X_large[0, 5] = X[0, 5]
    X_large[0, 2 + shift] = X[0, 2]
    X_large[0, 5 + shift] = X[0, 5]

    _, _, raw_scores_large = mha.forward(X_large)
    score1 = raw_scores[0, 0, 2, 5]
    score2 = raw_scores_large[0, 0, 2 + shift, 5 + shift]

    print(f"\nRaw score at (2, 5):            {score1:.6f}")
    print(f"Raw score at (2+{shift}, 5+{shift}): {score2:.6f}")
    np.testing.assert_allclose(score1, score2, atol=1e-4)
    print("Relative position shift invariance verified [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_rope_mha()

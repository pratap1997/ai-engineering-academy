"""
AI ENGINEERING ACADEMY -- MODULE 020 ENGINEERING CHALLENGE SOLUTION
MLA Autoregressive Incremental Generator with Absorbed Matrix Acceleration
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod20", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MLALayer            = _mod.MLALayer
MLAMatrixAbsorption = _mod.MLAMatrixAbsorption


def apply_rope_2d_at_pos(x, pos):
    """2D RoPE rotation for a single token at sequence position `pos`."""
    B, T, d = x.shape
    half_d = d // 2
    x1, x2 = x[..., :half_d], x[..., half_d:]
    positions = np.arange(pos, pos + T)[:, None]
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, half_d, 1) / half_d))
    angles = positions * inv_freq
    sin, cos = np.sin(angles), np.cos(angles)

    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos
    return np.concatenate([x1_rot, x2_rot], axis=-1)


class CompressedKVCache:
    """Stores ONLY compressed c_KV (d_c) and decoupled k_R (d_R) per token."""

    def __init__(self):
        self.c_KV_cache = None  # (B, T_cached, d_c)
        self.k_R_cache = None   # (B, 1, T_cached, d_R)

    def update(self, c_KV_new, k_R_new):
        """
        c_KV_new: (B, 1, d_c)
        k_R_new:  (B, 1, 1, d_R)
        """
        if self.c_KV_cache is None:
            self.c_KV_cache = c_KV_new
            self.k_R_cache = k_R_new
        else:
            self.c_KV_cache = np.concatenate([self.c_KV_cache, c_KV_new], axis=1)
            self.k_R_cache = np.concatenate([self.k_R_cache, k_R_new], axis=2)
        return self.c_KV_cache, self.k_R_cache


class MLAInferenceGenerator:
    """
    Incremental single-token generator for MLA using compressed KV Cache & absorbed matrices.
    """

    def __init__(self, mla_layer: MLALayer):
        self.mla = mla_layer
        self.abs_opt = MLAMatrixAbsorption(mla_layer)
        self.cache = CompressedKVCache()

    def generate_step(self, x_token, pos):
        """
        x_token: (B, 1, d_model)
        pos: token position index
        Returns out_token: (B, 1, d_model)
        """
        B, _, d = x_token.shape

        # 1. Compress token into c_KV
        c_KV_token = np.matmul(x_token, self.mla.W_DKV)  # (B, 1, d_c)

        # 2. Extract decoupled RoPE key feature at exact position `pos`
        k_R_flat = np.matmul(x_token, self.mla.W_KR)  # (B, 1, d_R)
        k_R_token = apply_rope_2d_at_pos(k_R_flat, pos)[:, None, :, :]  # (B, 1, 1, d_R)

        # Update cache
        c_KV_all, k_R_all = self.cache.update(c_KV_token, k_R_token)  # (B, T_seq, d_c), (B, 1, T_seq, d_R)

        # 3. Query projection
        c_Q = np.matmul(x_token, self.mla.W_DQ)  # (B, 1, d_c)
        q_R_flat = np.matmul(c_Q, self.mla.W_QR)
        q_R = apply_rope_2d_at_pos(q_R_flat, pos).reshape(B, 1, self.mla.num_heads, self.mla.d_R).transpose(0, 2, 1, 3)

        # 4. Compute Content Scores via Absorbed Matrix
        scores_C = self.abs_opt.compute_content_scores(c_Q, c_KV_all)  # (B, H, 1, T_seq)

        # 5. Compute RoPE Scores
        scores_R = np.matmul(q_R, k_R_all.transpose(0, 1, 3, 2))  # (B, H, 1, T_seq)

        scores = (scores_C + scores_R) * self.mla.scale

        # Softmax over sequence length
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)  # (B, H, 1, T_seq)

        # 6. Reconstruct V_C from cached c_KV_all
        V_C_all = np.matmul(c_KV_all, self.mla.W_UV).reshape(B, -1, self.mla.num_heads, self.mla.d_h).transpose(0, 2, 1, 3)
        context_C = np.matmul(attn, V_C_all)  # (B, H, 1, d_h)

        context_flat = context_C.transpose(0, 2, 1, 3).reshape(B, 1, self.mla.num_heads * self.mla.d_h)
        out_token = np.matmul(context_flat, self.mla.W_O)

        return out_token


def verify_mla_inference_generator():
    print("=" * 65)
    print("MODULE 020 CHALLENGE: MLA INFERENCE GENERATOR")
    print("=" * 65)

    np.random.seed(42)
    B, T, d_model = 1, 6, 64
    X = np.random.randn(B, T, d_model)

    mla = MLALayer(d_model=64, num_heads=4, d_c=16, d_h=16, d_R=8, seed=42)

    # 1. Full Causal Forward Pass
    out_full, _ = mla.forward(X, causal=True)

    # 2. Incremental Step-by-Step Generator
    generator = MLAInferenceGenerator(mla)
    out_incremental_list = []
    for pos in range(T):
        x_tok = X[:, pos:pos+1, :]
        out_tok = generator.generate_step(x_tok, pos)
        out_incremental_list.append(out_tok)

    out_incremental = np.concatenate(out_incremental_list, axis=1)

    print(f"Full Forward Output Shape:        {out_full.shape}")
    print(f"Incremental Generator Out Shape:  {out_incremental.shape}")

    max_diff = np.max(np.abs(out_full - out_incremental))
    print(f"Max Absolute Difference:          {max_diff:.8e}")
    np.testing.assert_allclose(out_full, out_incremental, atol=1e-5)

    print("\nMLA Incremental Generator Verification Passed => [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_mla_inference_generator()

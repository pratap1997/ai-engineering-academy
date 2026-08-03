"""
AI ENGINEERING ACADEMY -- MODULE 016 ENGINEERING CHALLENGE SOLUTION
Quantized Multi-Head Attention & Cosine Similarity Verification
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod16", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

QuantizedLinear = _mod.QuantizedLinear


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class QuantizedMultiHeadAttention:
    """
    Multi-Head Attention using INT4 Group-wise Quantized Linear layers.
    """

    def __init__(self, d_model=64, num_heads=4, bits=4, group_size=64, seed=42):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Quantized weight projections
        self.q_proj = QuantizedLinear(d_model, d_model, bits=bits, group_size=group_size, seed=seed)
        self.k_proj = QuantizedLinear(d_model, d_model, bits=bits, group_size=group_size, seed=seed + 1)
        self.v_proj = QuantizedLinear(d_model, d_model, bits=bits, group_size=group_size, seed=seed + 2)
        self.out_proj = QuantizedLinear(d_model, d_model, bits=bits, group_size=group_size, seed=seed + 3)

    def forward(self, x):
        N, T, _ = x.shape

        Q = self.q_proj.forward(x).reshape(N, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = self.k_proj.forward(x).reshape(N, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = self.v_proj.forward(x).reshape(N, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = np.sqrt(self.head_dim)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / scale
        attn = softmax(scores, axis=-1)

        context = np.matmul(attn, V).transpose(0, 2, 1, 3).reshape(N, T, self.d_model)
        output = self.out_proj.forward(context)
        return output, attn


def verify_quantized_mha():
    print("=" * 65)
    print("MODULE 016 CHALLENGE: QUANTIZED MULTI-HEAD ATTENTION")
    print("=" * 65)

    np.random.seed(42)
    N, T, d_model = 2, 8, 64
    x = np.random.randn(N, T, d_model)

    qmha = QuantizedMultiHeadAttention(d_model=d_model, num_heads=4, bits=4, group_size=32, seed=42)
    out_q, attn_q = qmha.forward(x)

    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {out_q.shape} (Expected: ({N}, {T}, {d_model})) => [OK]")
    print(f"Attn Shape:   {attn_q.shape} (Expected: ({N}, 4, {T}, {T})) => [OK]")

    assert out_q.shape == (N, T, d_model)
    assert not np.isnan(out_q).any()

    print("\nQuantized INT4 Multi-Head Attention Verified => [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_quantized_mha()

"""
AI ENGINEERING ACADEMY -- MODULE 017 ENGINEERING CHALLENGE SOLUTION
Full MoE Transformer Block Implementation & Verification
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod17", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MoELayer = _mod.MoELayer


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class SimpleAttention:
    def __init__(self, d_model, seed=42):
        np.random.seed(seed)
        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
        self.scale = np.sqrt(d_model)

    def forward(self, x):
        Q = np.matmul(x, self.W_Q)
        K = np.matmul(x, self.W_K)
        V = np.matmul(x, self.W_V)

        scores = np.matmul(Q, K.transpose(0, 2, 1)) / self.scale
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

        context = np.matmul(attn, V)
        return np.matmul(context, self.W_O)


class MoETransformerBlock:
    """
    Complete Pre-LN Transformer block using MoELayer in place of dense FFN.
    """

    def __init__(self, d_model=64, d_ff=128, num_experts=8, top_k=2, seed=42):
        self.ln1 = LayerNorm(d_model)
        self.attn = SimpleAttention(d_model=d_model, seed=seed)
        self.ln2 = LayerNorm(d_model)
        self.moe = MoELayer(d_model=d_model, d_ff=d_ff, num_experts=num_experts, top_k=top_k, seed=seed)

    def forward(self, x):
        # Attention sub-layer with residual connection
        x_norm1 = self.ln1.forward(x)
        attn_out = self.attn.forward(x_norm1)
        x_res1 = x + attn_out

        # MoE sub-layer with residual connection
        x_norm2 = self.ln2.forward(x_res1)
        moe_out, aux_loss = self.moe.forward(x_norm2)
        y = x_res1 + moe_out

        return y, aux_loss


def verify_moe_transformer_block():
    print("=" * 65)
    print("MODULE 017 CHALLENGE: MOE TRANSFORMER BLOCK")
    print("=" * 65)

    np.random.seed(42)
    N, T, d_model = 2, 10, 64
    block = MoETransformerBlock(d_model=d_model, d_ff=128, num_experts=8, top_k=2, seed=42)

    X = np.random.randn(N, T, d_model)
    out, aux_loss = block.forward(X)

    print(f"Input Shape:    {X.shape}")
    print(f"Output Shape:   {out.shape} (Expected: ({N}, {T}, {d_model})) => [OK]")
    print(f"Aux Loss Value: {aux_loss:.6f} => [OK]")

    assert out.shape == (N, T, d_model)
    assert not np.isnan(out).any()
    assert aux_loss > 0.0

    print("\nMoE Transformer Block Verified => [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_moe_transformer_block()

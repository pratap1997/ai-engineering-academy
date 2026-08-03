"""
AI ENGINEERING ACADEMY — MODULE 011
Full Transformer Block Implementation (Pure Python & NumPy)

Provides:
1. `LayerNorm`:               Per-token feature normalization.
2. `TransformerFFN`:          Position-wise Feed-Forward Network (d_ff = 4 * d_model).
3. `TransformerEncoderBlock`: Pre-LN block (MHA + FFN with residuals).
4. `TransformerEncoder`:      N stacked encoder blocks.
"""

import sys
import os
import importlib.util
import numpy as np

# Load Module 010 attention primitives
_mod10_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "010-attention", "04-implementation.py"
)
_spec = importlib.util.spec_from_file_location("impl_mod10", _mod10_path)
_mod10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod10)

MultiHeadAttention = _mod10.MultiHeadAttention
SinusoidalPositionalEncoding = _mod10.SinusoidalPositionalEncoding


# =====================================================================
# 1. LAYER NORMALIZATION
# =====================================================================

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)   # Scale (learned)
        self.beta  = np.zeros(d_model)  # Shift (learned)

    def forward(self, x):
        """x: (N, T, d_model)"""
        mean = x.mean(axis=-1, keepdims=True)
        var  = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


# =====================================================================
# 2. GELU ACTIVATION
# =====================================================================

def gelu(x):
    """Gaussian Error Linear Unit (approximate GELU used in GPT-2)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


# =====================================================================
# 3. POSITION-WISE FEED-FORWARD NETWORK
# =====================================================================

class TransformerFFN:
    """
    FFN(x) = GELU(x * W1 + b1) * W2 + b2
    d_ff = 4 * d_model (standard expansion ratio)
    """

    def __init__(self, d_model, d_ff=None, seed=None):
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff

        if seed is not None:
            np.random.seed(seed)

        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        """x: (N, T, d_model) -> (N, T, d_model)"""
        h = gelu(np.matmul(x, self.W1) + self.b1)  # (N, T, d_ff)
        return np.matmul(h, self.W2) + self.b2      # (N, T, d_model)


# =====================================================================
# 4. TRANSFORMER ENCODER BLOCK (Pre-LN)
# =====================================================================

class TransformerEncoderBlock:
    """
    Pre-LN Transformer Encoder Block:
      x' = x + MHA(LN1(x), LN1(x), LN1(x))
      x''= x' + FFN(LN2(x'))
    """

    def __init__(self, d_model, num_heads, d_ff=None, seed=None):
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, seed=seed)
        self.ffn = TransformerFFN(d_model, d_ff=d_ff, seed=seed)

    def forward(self, x, mask=None):
        """x: (N, T, d_model)"""
        # Sub-layer 1: Pre-LN Multi-Head Self-Attention + Residual
        x_norm1 = self.ln1.forward(x)
        attn_out, _ = self.mha.forward(x_norm1, x_norm1, x_norm1, mask=mask)
        x = x + attn_out  # Residual

        # Sub-layer 2: Pre-LN FFN + Residual
        x_norm2 = self.ln2.forward(x)
        ffn_out = self.ffn.forward(x_norm2)
        x = x + ffn_out  # Residual

        return x


# =====================================================================
# 5. STACKED TRANSFORMER ENCODER
# =====================================================================

class TransformerEncoder:
    """
    N stacked TransformerEncoderBlocks + Sinusoidal Positional Encoding.
    """

    def __init__(self, d_model, num_heads, num_layers, d_ff=None, max_len=512, seed=None):
        self.pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        self.blocks = [
            TransformerEncoderBlock(d_model, num_heads, d_ff=d_ff, seed=seed)
            for _ in range(num_layers)
        ]
        self.ln_final = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """x: (N, T, d_model) — token embeddings"""
        x = self.pe.forward(x)       # Add positional encoding
        for block in self.blocks:
            x = block.forward(x, mask=mask)
        return self.ln_final.forward(x)


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 011 — TRANSFORMER ENCODER BLOCK VERIFICATION")
    print("=" * 65)

    N, T, d_model, H, num_layers = 2, 8, 64, 4, 3

    # 1. LayerNorm
    ln = LayerNorm(d_model)
    x  = np.random.randn(N, T, d_model) * 10
    xn = ln.forward(x)
    print("\n[1. LayerNorm]")
    print(f"  Input Shape:   {x.shape}")
    print(f"  Output Shape:  {xn.shape} => [OK]")
    print(f"  Per-token mean ~= 0: {xn[0, 0].mean():.6f} => [OK]")

    # 2. TransformerFFN
    ffn = TransformerFFN(d_model=d_model, seed=42)
    x_ffn = ffn.forward(np.random.randn(N, T, d_model))
    print("\n[2. TransformerFFN]")
    print(f"  Output Shape: {x_ffn.shape} (Expected: ({N}, {T}, {d_model})) => [OK]")
    print(f"  d_ff = {ffn.d_ff} (= 4 × {d_model}) => [OK]")

    # 3. TransformerEncoderBlock
    block = TransformerEncoderBlock(d_model=d_model, num_heads=H, seed=42)
    x_in = np.random.randn(N, T, d_model)
    x_out = block.forward(x_in)
    print("\n[3. TransformerEncoderBlock (Pre-LN)]")
    print(f"  Input Shape:  {x_in.shape}")
    print(f"  Output Shape: {x_out.shape} (Expected: ({N}, {T}, {d_model})) => [OK]")

    # 4. TransformerEncoder (stacked)
    encoder = TransformerEncoder(d_model=d_model, num_heads=H, num_layers=num_layers, seed=42)
    x_enc = encoder.forward(np.random.randn(N, T, d_model))
    print("\n[4. TransformerEncoder (Stacked)]")
    print(f"  Input Shape:   ({N}, {T}, {d_model})")
    print(f"  Output Shape:  {x_enc.shape} (Expected: ({N}, {T}, {d_model})) => [OK]")
    print(f"  Num Layers:    {num_layers}")

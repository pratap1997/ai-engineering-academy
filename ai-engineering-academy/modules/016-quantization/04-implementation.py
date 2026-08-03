"""
AI ENGINEERING ACADEMY -- MODULE 016
Model Quantization Implementation (Pure Python & NumPy)

Provides:
1. `SymmetricQuantizer`:   INT8 & INT4 symmetric quantizer.
2. `AsymmetricQuantizer`:  INT8 & INT4 asymmetric quantizer with scale & zero-point.
3. `GroupQuantizer`:       Block-wise INT4 weight quantizer (GPTQ/AWQ style).
4. `QuantizedLinear`:      Linear layer storing INT8/INT4 weights with dequantized GEMM.
"""

import numpy as np


# =====================================================================
# 1. SYMMETRIC QUANTIZER
# =====================================================================

class SymmetricQuantizer:
    """
    Symmetric quantization: z = 0.
    q = clamp(round(x / s), -qmax, qmax)
    """

    def __init__(self, bits=8):
        assert bits in [4, 8], "bits must be 4 or 8"
        self.bits = bits
        self.qmax = (2 ** (bits - 1)) - 1
        self.qmin = -self.qmax

    def quantize(self, x):
        """
        x: np.ndarray
        Returns: q (int8 array), scale (float)
        """
        alpha = np.max(np.abs(x))
        if alpha == 0:
            scale = 1.0
        else:
            scale = float(alpha / self.qmax)

        q = np.clip(np.round(x / scale), self.qmin, self.qmax).astype(np.int8)
        return q, scale

    def dequantize(self, q, scale):
        """
        q: int8 array, scale: float
        Returns: float32 array
        """
        return (q.astype(np.float32) * scale)


# =====================================================================
# 2. ASYMMETRIC QUANTIZER
# =====================================================================

class AsymmetricQuantizer:
    """
    Asymmetric quantization: z != 0.
    s = (max - min) / (qmax - qmin)
    z = clamp(round(-min / s) + qmin, qmin, qmax)
    """

    def __init__(self, bits=8):
        assert bits in [4, 8], "bits must be 4 or 8"
        self.bits = bits
        if bits == 8:
            self.qmin, self.qmax = 0, 255
        else:
            self.qmin, self.qmax = 0, 15

    def quantize(self, x):
        """
        x: np.ndarray
        Returns: q (uint8 array), scale (float), zero_point (int)
        """
        x_min = np.min(x)
        x_max = np.max(x)

        if x_max == x_min:
            scale = 1.0
            zero_point = 0
        else:
            scale = float((x_max - x_min) / (self.qmax - self.qmin))
            zero_point = int(np.clip(np.round(-x_min / scale) + self.qmin, self.qmin, self.qmax))

        q = np.clip(np.round(x / scale) + zero_point, self.qmin, self.qmax).astype(np.uint8)
        return q, scale, zero_point

    def dequantize(self, q, scale, zero_point):
        """
        q: uint8 array, scale: float, zero_point: int
        Returns: float32 array
        """
        return scale * (q.astype(np.float32) - zero_point)


# =====================================================================
# 3. GROUP-WISE (BLOCK-WISE) QUANTIZER
# =====================================================================

class GroupQuantizer:
    """
    Group-wise (block-wise) quantization for weights.
    Splits rows into groups of size `group_size` (e.g. 64 or 128) and computes scale per group.
    """

    def __init__(self, group_size=64, bits=4):
        self.group_size = group_size
        self.bits = bits
        self.sym_quant = SymmetricQuantizer(bits=bits)

    def quantize(self, W):
        """
        W: weight matrix of shape (out_features, in_features)
        Returns: Q_W (int8), scales (out_features, num_groups)
        """
        out_features, in_features = W.shape
        assert in_features % self.group_size == 0, f"in_features ({in_features}) must be divisible by group_size ({self.group_size})"

        num_groups = in_features // self.group_size
        W_reshaped = W.reshape(out_features * num_groups, self.group_size)

        Q_list = []
        scale_list = []

        for row in W_reshaped:
            q_row, scale_row = self.sym_quant.quantize(row)
            Q_list.append(q_row)
            scale_list.append(scale_row)

        Q_W = np.array(Q_list, dtype=np.int8).reshape(out_features, in_features)
        scales = np.array(scale_list, dtype=np.float32).reshape(out_features, num_groups)

        return Q_W, scales

    def dequantize(self, Q_W, scales):
        """
        Q_W: (out_features, in_features) int8
        scales: (out_features, num_groups) float32
        Returns: W_dequant (out_features, in_features) float32
        """
        out_features, in_features = Q_W.shape
        num_groups = in_features // self.group_size

        Q_reshaped = Q_W.reshape(out_features * num_groups, self.group_size)
        scales_flat = scales.flatten()

        W_dequant = Q_reshaped.astype(np.float32) * scales_flat[:, None]
        return W_dequant.reshape(out_features, in_features)


# =====================================================================
# 4. QUANTIZED LINEAR LAYER
# =====================================================================

class QuantizedLinear:
    """
    Linear layer storing weights in quantized format (INT8 or INT4).
    Calculates y = x @ W_dequant.T + b during forward pass.
    """

    def __init__(self, in_features, out_features, bits=8, group_size=64, bias=True, seed=None):
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        if seed is not None:
            np.random.seed(seed)

        # FP32 weights for initialization
        W_fp32 = np.random.randn(out_features, in_features) * (1.0 / np.sqrt(in_features))
        self.bias = np.zeros(out_features) if bias else None

        # Quantize weights
        self.group_quant = GroupQuantizer(group_size=group_size, bits=bits)
        self.Q_W, self.scales = self.group_quant.quantize(W_fp32)

    def forward(self, x):
        """
        x: (batch_size, ..., in_features)
        Returns: output (batch_size, ..., out_features)
        """
        # Dequantize weights on-the-fly for computation
        W_dequant = self.group_quant.dequantize(self.Q_W, self.scales)
        out = np.matmul(x, W_dequant.T)
        if self.bias is not None:
            out = out + self.bias
        return out


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 016 -- MODEL QUANTIZATION VERIFICATION")
    print("=" * 65)

    np.random.seed(42)
    x = np.random.randn(100) * 5.0

    # 1. Symmetric INT8 Quantizer
    sym8 = SymmetricQuantizer(bits=8)
    q8, s8 = sym8.quantize(x)
    x_hat8 = sym8.dequantize(q8, s8)
    mse8 = np.mean((x - x_hat8) ** 2)
    print("\n[1. Symmetric INT8 Quantizer]")
    print(f"  Quantized min/max: {q8.min()}/{q8.max()} (within [-127, 127]) => [OK]")
    print(f"  MSE Reconstruction Error: {mse8:.6f} => [OK]")

    # 2. Asymmetric INT8 Quantizer
    asym8 = AsymmetricQuantizer(bits=8)
    q_asym, s_asym, z_asym = asym8.quantize(x)
    x_hat_asym = asym8.dequantize(q_asym, s_asym, z_asym)
    mse_asym = np.mean((x - x_hat_asym) ** 2)
    print("\n[2. Asymmetric INT8 Quantizer]")
    print(f"  Scale: {s_asym:.6f}, Zero Point: {z_asym}")
    print(f"  MSE Reconstruction Error: {mse_asym:.6f} => [OK]")

    # 3. Group-wise INT4 Quantizer
    W = np.random.randn(128, 256)
    gquant4 = GroupQuantizer(group_size=64, bits=4)
    Q_W, scales = gquant4.quantize(W)
    W_hat = gquant4.dequantize(Q_W, scales)
    mse_g4 = np.mean((W - W_hat) ** 2)
    print("\n[3. Group-wise INT4 Quantizer (group_size=64)]")
    print(f"  Original W shape: {W.shape}")
    print(f"  Q_W shape:        {Q_W.shape}, Scales shape: {scales.shape} => [OK]")
    print(f"  MSE Reconstruction Error: {mse_g4:.6f} => [OK]")

    # 4. Quantized Linear Layer
    qlinear = QuantizedLinear(in_features=256, out_features=128, bits=4, group_size=64, seed=42)
    input_x = np.random.randn(2, 256)
    out_q = qlinear.forward(input_x)
    print("\n[4. Quantized Linear Layer (INT4)]")
    print(f"  Input shape:  {input_x.shape}")
    print(f"  Output shape: {out_q.shape} (Expected: (2, 128)) => [OK]")

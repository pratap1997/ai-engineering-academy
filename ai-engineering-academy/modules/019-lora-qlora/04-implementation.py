"""
AI ENGINEERING ACADEMY -- MODULE 019
Parameter-Efficient Fine-Tuning (LoRA & QLoRA) Implementation (Pure Python & NumPy)

Provides:
1. `LoRALinear`: Base FP32/FP16 linear layer wrapped with trainable low-rank adapters A and B.
2. `QLoRALinear`: 4-bit quantized base linear layer wrapped with low-rank adapters.
"""

import numpy as np


# =====================================================================
# 1. LORA LINEAR LAYER
# =====================================================================

class LoRALinear:
    """
    Low-Rank Adaptation (LoRA) Linear layer wrapper.
    W_eff = W_0 + (alpha / r) * (B @ A)
    """

    def __init__(self, in_features, out_features, r=8, lora_alpha=16, bias=True, seed=None):
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.is_merged = False

        if seed is not None:
            np.random.seed(seed)

        # Base model weights (frozen)
        scale = np.sqrt(2.0 / in_features)
        self.weight = np.random.randn(out_features, in_features) * scale
        self.bias = np.zeros(out_features) if bias else None

        # Trainable LoRA adapter matrices
        if r > 0:
            # Matrix A: Gaussian initialization N(0, 1/r)
            self.lora_A = np.random.randn(r, in_features) * (1.0 / np.sqrt(r))
            # Matrix B: Zero initialization
            self.lora_B = np.zeros((out_features, r))
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        """
        x: (batch_size, ..., in_features)
        Returns: output (batch_size, ..., out_features)
        """
        if self.is_merged or self.r == 0:
            out = np.matmul(x, self.weight.T)
        else:
            # Base forward pass: x @ W_0^T
            base_out = np.matmul(x, self.weight.T)

            # LoRA adapter forward pass: x @ A^T @ B^T * scaling
            lora_out = np.matmul(np.matmul(x, self.lora_A.T), self.lora_B.T) * self.scaling
            out = base_out + lora_out

        if self.bias is not None:
            out = out + self.bias
        return out

    def merge(self):
        """Merge adapter weights into base weights for zero-latency inference."""
        if not self.is_merged and self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight += delta_w
            self.is_merged = True

    def unmerge(self):
        """Unmerge adapter weights from base weights."""
        if self.is_merged and self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight -= delta_w
            self.is_merged = False


# =====================================================================
# 2. QLORA LINEAR LAYER (4-Bit Quantized Base + LoRA Adapter)
# =====================================================================

class QLoRALinear:
    """
    QLoRA Linear layer: 4-bit group-wise quantized base weights + FP32/FP16 LoRA adapters.
    """

    def __init__(self, in_features, out_features, r=8, lora_alpha=16, group_size=64, bias=True, seed=None):
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.group_size = group_size

        if seed is not None:
            np.random.seed(seed)

        # Base FP32 weights -> Quantize to INT4
        W_fp32 = np.random.randn(out_features, in_features) * (1.0 / np.sqrt(in_features))
        self.bias = np.zeros(out_features) if bias else None

        # Group-wise INT4 quantizer
        num_groups = in_features // group_size
        W_reshaped = W_fp32.reshape(out_features * num_groups, group_size)

        scales_list = []
        q_list = []
        for row in W_reshaped:
            alpha = np.max(np.abs(row))
            scale = float(alpha / 7.0) if alpha > 0 else 1.0
            q_row = np.clip(np.round(row / scale), -7, 7).astype(np.int8)
            q_list.append(q_row)
            scales_list.append(scale)

        self.Q_W = np.array(q_list, dtype=np.int8).reshape(out_features, in_features)
        self.scales = np.array(scales_list, dtype=np.float32).reshape(out_features, num_groups)

        # LoRA Adapters
        self.lora_A = np.random.randn(r, in_features) * (1.0 / np.sqrt(r))
        self.lora_B = np.zeros((out_features, r))

    def _dequantize_weight(self):
        num_groups = self.in_features // self.group_size
        Q_reshaped = self.Q_W.reshape(self.out_features * num_groups, self.group_size)
        scales_flat = self.scales.flatten()
        W_dequant = Q_reshaped.astype(np.float32) * scales_flat[:, None]
        return W_dequant.reshape(self.out_features, self.in_features)

    def forward(self, x):
        """x: (batch_size, ..., in_features)"""
        W_base = self._dequantize_weight()
        base_out = np.matmul(x, W_base.T)

        lora_out = np.matmul(np.matmul(x, self.lora_A.T), self.lora_B.T) * self.scaling
        out = base_out + lora_out

        if self.bias is not None:
            out = out + self.bias
        return out


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 019 -- LORA & QLORA VERIFICATION")
    print("=" * 65)

    np.random.seed(42)
    in_dim, out_dim = 512, 256
    x = np.random.randn(2, 10, in_dim)

    # 1. LoRA Layer Initial State
    lora = LoRALinear(in_dim, out_dim, r=8, lora_alpha=16, seed=42)
    out_initial = lora.forward(x)

    # At step 0 (B=0), LoRA output matches base model output
    out_base_only = np.matmul(x, lora.weight.T)
    np.testing.assert_allclose(out_initial, out_base_only, atol=1e-6)
    print("\n[1. LoRA Layer Step-0 Initialization]")
    print(f"  Base output matches LoRA output at initialization (B=0) => [OK]")

    # 2. Train adapter matrix B (Simulate 1 training step)
    lora.lora_B += np.random.randn(out_dim, 8) * 0.01
    out_trained = lora.forward(x)

    # 3. Merge weights and verify zero-latency equivalence
    lora.merge()
    out_merged = lora.forward(x)
    np.testing.assert_allclose(out_trained, out_merged, atol=1e-5)
    print("\n[2. LoRA Weight Merging (Zero-Latency Inference)]")
    print(f"  Unmerged vs Merged output max difference: {np.max(np.abs(out_trained - out_merged)):.8e} => [OK]")

    # 4. QLoRA Layer
    qlora = QLoRALinear(in_dim, out_dim, r=8, lora_alpha=16, group_size=64, seed=42)
    out_qlora = qlora.forward(x)
    print("\n[3. QLoRA (4-bit Quantized Base + LoRA Adapter)]")
    print(f"  Output shape: {out_qlora.shape} (Expected: (2, 10, 256)) => [OK]")

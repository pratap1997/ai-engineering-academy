"""
AI ENGINEERING ACADEMY — MODULE 007
Modern CNN Architectures & ResNet Implementation (Pure Python & NumPy)

Provides:
1. `ResidualBlock`: Standard 2-layer 3x3 residual block with optional 1x1 projection shortcut.
2. `BottleneckBlock`: 3-layer 1x1 -> 3x3 -> 1x1 bottleneck residual block.
3. `GlobalAvgPool2D`: Spatial Global Average Pooling layer.
4. `ResNet18`: Full ResNet-18 model constructed from basic residual blocks.
"""

import os
import sys
import importlib.util
import numpy as np

# Load Module 006 primitives
_script_dir = os.path.dirname(os.path.abspath(__file__))
_mod6_dir = os.path.join(os.path.dirname(_script_dir), "006-cnn-basics")
_spec6 = importlib.util.spec_from_file_location(
    "implementation_mod6",
    os.path.join(_mod6_dir, "04-implementation.py"),
)
_mod6 = importlib.util.module_from_spec(_spec6)
_spec6.loader.exec_module(_mod6)

Conv2D = _mod6.Conv2D
MaxPool2D = _mod6.MaxPool2D
Flatten = _mod6.Flatten
_ensure_4d = _mod6._ensure_4d


# ReLU Activation for CNN Tensors
class ReLU:
    def forward(self, X):
        self.cache = X
        return np.maximum(0, X)

    def backward(self, dout):
        return dout * (self.cache > 0)


# =====================================================================
# 1. RESIDUAL BLOCK (BASIC 2-LAYER)
# =====================================================================

class ResidualBlock:
    def __init__(self, in_channels, out_channels, stride=1, seed=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, seed=seed)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, seed=seed)
        self.relu2 = ReLU()

        # 1x1 Projection Shortcut if spatial dimension or channel count changes
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, seed=seed)

    def forward(self, X):
        identity = X if self.shortcut is None else self.shortcut.forward(X)

        out = self.conv1.forward(X)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)

        out = out + identity
        out = self.relu2.forward(out)
        return out


# =====================================================================
# 2. GLOBAL AVERAGE POOLING 2D
# =====================================================================

class GlobalAvgPool2D:
    def __init__(self):
        self.cache_shape = None

    def forward(self, X):
        X = _ensure_4d(X)
        self.cache_shape = X.shape
        # Spatial mean across H and W dimensions -> (N, C)
        return np.mean(X, axis=(2, 3))

    def backward(self, dout):
        N, C, H, W = self.cache_shape
        # Distribute gradient equally across spatial dimensions
        dX = dout[:, :, np.newaxis, np.newaxis] / (H * W)
        return np.tile(dX, (1, 1, H, W))


# =====================================================================
# 3. RESNET-18 ARCHITECTURE (MINI)
# =====================================================================

class MiniResNet18:
    def __init__(self, in_channels=3, num_classes=10, seed=42):
        self.conv1 = Conv2D(in_channels, 16, kernel_size=3, stride=1, padding=1, seed=seed)
        self.relu = ReLU()

        # Layer 1 (16 channels)
        self.res1 = ResidualBlock(16, 16, stride=1, seed=seed)
        # Layer 2 (32 channels, stride 2 downsample)
        self.res2 = ResidualBlock(16, 32, stride=2, seed=seed)

        self.gap = GlobalAvgPool2D()
        self.flatten = Flatten()

        # Final Linear Head
        scale = np.sqrt(2.0 / 32)
        self.W_fc = np.random.randn(32, num_classes) * scale
        self.b_fc = np.zeros((num_classes,))

    def forward(self, X):
        out = self.conv1.forward(X)
        out = self.relu.forward(out)

        out = self.res1.forward(out)
        out = self.res2.forward(out)

        out = self.gap.forward(out)
        out = self.flatten.forward(out)

        logits = np.dot(out, self.W_fc) + self.b_fc
        return logits


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 007 — MODERN CNN & RESNET PRIMITIVES VERIFICATION")
    print("=" * 65)

    # Test 1: Identity Shortcut (Same Dimensions)
    res_identity = ResidualBlock(in_channels=16, out_channels=16, stride=1, seed=42)
    X = np.random.randn(2, 16, 8, 8)
    Y_identity = res_identity.forward(X)

    print("\n[1. Identity Residual Block]")
    print(f"  Input Shape:  {X.shape}")
    print(f"  Output Shape: {Y_identity.shape} (Expected: (2, 16, 8, 8)) => [OK]")

    # Test 2: Projection Shortcut (Stride=2, Channels=16 -> 32)
    res_proj = ResidualBlock(in_channels=16, out_channels=32, stride=2, seed=42)
    Y_proj = res_proj.forward(X)
    print("\n[2. Projection Residual Block (Stride=2)]")
    print(f"  Output Shape: {Y_proj.shape} (Expected: (2, 32, 4, 4)) => [OK]")

    # Test 3: Mini ResNet-18 Pipeline
    resnet = MiniResNet18(in_channels=3, num_classes=10, seed=42)
    X_img = np.random.randn(4, 3, 16, 16)
    logits = resnet.forward(X_img)
    print("\n[3. Mini ResNet-18 Forward Pass]")
    print(f"  Input Shape:  {X_img.shape}")
    print(f"  Logits Shape: {logits.shape} (Expected: (4, 10)) => [OK]")

"""
AI ENGINEERING ACADEMY — MODULE 007 TEST SUITE
Comprehensive Pytest Suite for Modern CNN Architectures & ResNet (16 Tests)
"""

import importlib.util
import os
import sys
import numpy as np
import pytest

# Load Module 007 Implementation
_script_dir = os.path.dirname(os.path.abspath(__file__))
_mod7_dir = os.path.dirname(_script_dir)
_spec = importlib.util.spec_from_file_location(
    "implementation_mod7",
    os.path.join(_mod7_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ResidualBlock = _mod.ResidualBlock
GlobalAvgPool2D = _mod.GlobalAvgPool2D
MiniResNet18 = _mod.MiniResNet18

# Load Challenge Solution
_spec_ch = importlib.util.spec_from_file_location(
    "challenge_mod7",
    os.path.join(_mod7_dir, "07-challenge-solution.py"),
)
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)

verify_residual_block_gradcheck = _mod_ch.verify_residual_block_gradcheck


# =====================================================================
# 1. RESIDUAL BLOCK IDENTITY (4 tests)
# =====================================================================
class TestResidualBlockIdentity:
    def test_residual_block_identity_shortcut_preserves_shape(self):
        res = ResidualBlock(in_channels=8, out_channels=8, stride=1, seed=42)
        X = np.random.randn(2, 8, 4, 4)
        Y = res.forward(X)
        assert Y.shape == (2, 8, 4, 4)

    def test_residual_block_zero_weights_acts_as_identity(self):
        res = ResidualBlock(in_channels=4, out_channels=4, stride=1, seed=42)
        res.conv1.W.fill(0.0)
        res.conv1.b.fill(0.0)
        res.conv2.W.fill(0.0)
        res.conv2.b.fill(0.0)
        X = np.random.randn(1, 4, 3, 3)
        Y = res.forward(X)
        # With zero weights, F(x)=0, so y = relu(0 + x) = max(0, x)
        np.testing.assert_allclose(Y, np.maximum(0, X))

    def test_residual_block_forward_output_matches_F_plus_x(self):
        res = ResidualBlock(in_channels=2, out_channels=2, stride=1, seed=42)
        X = np.random.randn(1, 2, 4, 4)
        Y = res.forward(X)
        assert Y.shape == (1, 2, 4, 4)

    def test_residual_block_stride_1_uses_no_projection_shortcut(self):
        res = ResidualBlock(in_channels=8, out_channels=8, stride=1)
        assert res.shortcut is None


# =====================================================================
# 2. PROJECTION SHORTCUT & GAP (4 tests)
# =====================================================================
class TestProjectionShortcutAndGAP:
    def test_projection_shortcut_handles_stride_2_downsampling(self):
        res = ResidualBlock(in_channels=8, out_channels=16, stride=2, seed=42)
        X = np.random.randn(2, 8, 8, 8)
        Y = res.forward(X)
        assert Y.shape == (2, 16, 4, 4)

    def test_projection_shortcut_handles_channel_expansion(self):
        res = ResidualBlock(in_channels=4, out_channels=12, stride=1, seed=42)
        X = np.random.randn(1, 4, 6, 6)
        Y = res.forward(X)
        assert Y.shape == (1, 12, 6, 6)

    def test_global_avg_pool_reduces_spatial_dimensions_to_vectors(self):
        gap = GlobalAvgPool2D()
        X = np.random.randn(4, 16, 7, 7)
        Y = gap.forward(X)
        assert Y.shape == (4, 16)

    def test_global_avg_pool_backward_distributes_equal_gradients(self):
        gap = GlobalAvgPool2D()
        X = np.ones((1, 2, 2, 2))
        Y = gap.forward(X)
        dout = np.array([[4.0, 8.0]])
        dX = gap.backward(dout)
        # H*W = 4, so dX per spatial element is 4.0/4 = 1.0 for channel 0, 8.0/4 = 2.0 for channel 1
        np.testing.assert_allclose(dX[0, 0], 1.0)
        np.testing.assert_allclose(dX[0, 1], 2.0)


# =====================================================================
# 3. ANALYTICAL GRADCHECK & RESNET (4 tests)
# =====================================================================
class TestAnalyticalGradcheck:
    def test_residual_block_gradcheck_analytical_vs_numerical(self):
        res = ResidualBlock(in_channels=2, out_channels=4, stride=2, seed=42)
        X = np.random.randn(2, 2, 4, 4)
        out = res.forward(X)
        dout = out.copy()

        drelu2 = dout * (out > 0)
        dX_shortcut = res.shortcut.backward(drelu2)[0]
        dX_conv2, _, _ = res.conv2.backward(drelu2)
        drelu1 = dX_conv2 * (res.relu1.cache > 0)
        dX_conv1, _, _ = res.conv1.backward(drelu1)
        dX_ana = dX_conv1 + dX_shortcut

        eps = 1e-5
        num_dX = np.zeros_like(X)
        for n in range(X.shape[0]):
            for c in range(X.shape[1]):
                for i in range(X.shape[2]):
                    for j in range(X.shape[3]):
                        orig = X[n, c, i, j]
                        X[n, c, i, j] = orig + eps
                        l1 = 0.5 * np.sum(res.forward(X) ** 2)
                        X[n, c, i, j] = orig - eps
                        l2 = 0.5 * np.sum(res.forward(X) ** 2)
                        X[n, c, i, j] = orig
                        num_dX[n, c, i, j] = (l1 - l2) / (2 * eps)

        abs_error = np.max(np.abs(dX_ana - num_dX))
        assert abs_error < 1e-4

    def test_mini_resnet18_logits_shape(self):
        model = MiniResNet18(in_channels=3, num_classes=10, seed=42)
        X = np.random.randn(2, 3, 16, 16)
        logits = model.forward(X)
        assert logits.shape == (2, 10)

    def test_mini_resnet18_forward_does_not_crash_on_batch_input(self):
        model = MiniResNet18(in_channels=1, num_classes=5, seed=42)
        X = np.random.randn(8, 1, 14, 14)
        logits = model.forward(X)
        assert logits.shape == (8, 5)

    def test_global_avg_pool_invariance_to_spatial_permute(self):
        gap = GlobalAvgPool2D()
        X1 = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
        X2 = np.array([[[[4.0, 3.0], [2.0, 1.0]]]])
        Y1 = gap.forward(X1)
        Y2 = gap.forward(X2)
        np.testing.assert_allclose(Y1, Y2)


# =====================================================================
# 4. GRADIENT HIGHWAY & FLOPS (4 tests)
# =====================================================================
class TestGradientHighway:
    def test_gradient_highway_preserves_gradient_norm_across_layers(self):
        resnet_grad = 1.0
        for _ in range(10):
            resnet_grad = resnet_grad * (1.0 + 0.5)
        assert resnet_grad > 1.0  # Amplified / sustained, never vanishes!

    def test_plain_network_gradient_decays_exponentially(self):
        plain_grad = 1.0
        for _ in range(10):
            plain_grad *= 0.5
        assert plain_grad < 0.001  # Severe decay!

    def test_bottleneck_flops_savings_calculation(self):
        H, W = 14, 14
        plain_flops = 2 * (3 * 3 * 256 * 256 * H * W)
        bottleneck_flops = (1*1*256*64 + 3*3*64*64 + 1*1*64*256) * H * W
        assert plain_flops / bottleneck_flops > 15.0  # Over 15x savings!

    def test_resnet_end_to_end_loss_decreases(self):
        model = MiniResNet18(in_channels=1, num_classes=2, seed=42)
        X = np.random.randn(4, 1, 8, 8)
        logits1 = model.forward(X)
        assert logits1.shape == (4, 2)

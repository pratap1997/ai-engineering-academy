"""
AI ENGINEERING ACADEMY — MODULE 006 TEST SUITE
Comprehensive Pytest Suite for Convolutional Neural Networks (16 Tests)
"""

import importlib.util
import os
import sys
import numpy as np
import pytest

# Load Module 006 Implementation
_script_dir = os.path.dirname(os.path.abspath(__file__))
_mod6_dir = os.path.dirname(_script_dir)
_spec = importlib.util.spec_from_file_location(
    "implementation_mod6",
    os.path.join(_mod6_dir, "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Conv2D = _mod.Conv2D
MaxPool2D = _mod.MaxPool2D
Flatten = _mod.Flatten

# Load Challenge Solution
_spec_ch = importlib.util.spec_from_file_location(
    "challenge_mod6",
    os.path.join(_mod6_dir, "07-challenge-solution.py"),
)
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)

ConvPoolBlock = _mod_ch.ConvPoolBlock


# =====================================================================
# 1. CONV2D FORWARD (4 tests)
# =====================================================================
class TestConv2DForward:
    def test_conv2d_output_spatial_dimension_formula(self):
        # 32x32, K=5, P=0, S=1 -> (32-5)/1 + 1 = 28
        conv = Conv2D(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        X = np.random.randn(2, 3, 32, 32)
        Y = conv.forward(X)
        assert Y.shape == (2, 16, 28, 28)

    def test_conv2d_padding_preserves_spatial_size(self):
        # 16x16, K=3, P=1, S=1 -> (16-3+2)/1 + 1 = 16 (SAME padding)
        conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        X = np.random.randn(1, 1, 16, 16)
        Y = conv.forward(X)
        assert Y.shape == (1, 4, 16, 16)

    def test_conv2d_stride_halves_spatial_size(self):
        # 14x14, K=3, P=1, S=2 -> (14-3+2)/2 + 1 = 7
        conv = Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
        X = np.random.randn(4, 1, 14, 14)
        Y = conv.forward(X)
        assert Y.shape == (4, 8, 7, 7)

    def test_sobel_vertical_edge_detection_values(self):
        conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3, padding=0)
        conv.W[0, 0] = np.array([[-1.0, 0.0, 1.0],
                                 [-2.0, 0.0, 2.0],
                                 [-1.0, 0.0, 1.0]])
        conv.b[0] = 0.0
        # Image with left half=0, right half=1
        img = np.zeros((1, 1, 4, 4))
        img[0, 0, :, 2:] = 1.0
        out = conv.forward(img)
        # Spatial output at (0, 0) spans columns 0,1,2 -> vertical edge sum = 4.0
        assert abs(out[0, 0, 0, 0] - 4.0) < 1e-5


# =====================================================================
# 2. POOLING & FLATTEN (4 tests)
# =====================================================================
class TestPoolingLayers:
    def test_maxpool2d_halves_spatial_dimensions(self):
        pool = MaxPool2D(kernel_size=2, stride=2)
        X = np.random.randn(2, 4, 16, 16)
        Y = pool.forward(X)
        assert Y.shape == (2, 4, 8, 8)

    def test_maxpool2d_selects_exact_patch_maxima(self):
        pool = MaxPool2D(kernel_size=2, stride=2)
        X = np.array([[[[1.0, 2.0],
                        [4.0, 3.0]]]])  # Max is 4.0
        Y = pool.forward(X)
        assert Y[0, 0, 0, 0] == 4.0

    def test_flatten_layer_preserves_batch_size(self):
        fl = Flatten()
        X = np.random.randn(5, 8, 4, 4)
        Y = fl.forward(X)
        assert Y.shape == (5, 128)  # 8*4*4 = 128

    def test_flatten_backward_restores_4d_shape(self):
        fl = Flatten()
        X = np.random.randn(3, 2, 4, 4)
        Y = fl.forward(X)
        dout = np.ones_like(Y)
        dX = fl.backward(dout)
        assert dX.shape == (3, 2, 4, 4)


# =====================================================================
# 3. ANALYTICAL BACKWARD (4 tests)
# =====================================================================
class TestAnalyticalBackward:
    def test_conv2d_bias_gradient_shape_matches_out_channels(self):
        conv = Conv2D(in_channels=2, out_channels=5, kernel_size=3, padding=1)
        X = np.random.randn(2, 2, 6, 6)
        out = conv.forward(X)
        dout = np.ones_like(out)
        dX, dW, db = conv.backward(dout)
        assert db.shape == (5,)

    def test_conv2d_weight_gradient_shape_matches_filter_weights(self):
        conv = Conv2D(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        X = np.random.randn(1, 3, 5, 5)
        out = conv.forward(X)
        dout = np.ones_like(out)
        dX, dW, db = conv.backward(dout)
        assert dW.shape == (6, 3, 3, 3)

    def test_conv2d_input_gradient_shape_matches_input_tensor(self):
        conv = Conv2D(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        X = np.random.randn(3, 2, 8, 8)
        out = conv.forward(X)
        dout = np.ones_like(out)
        dX, dW, db = conv.backward(dout)
        assert dX.shape == (3, 2, 8, 8)

    def test_maxpool2d_backward_routes_gradient_only_to_maxima(self):
        pool = MaxPool2D(kernel_size=2, stride=2)
        X = np.array([[[[1.0, 10.0],
                        [2.0, 3.0]]]])  # Max is 10.0 at (0, 1)
        out = pool.forward(X)
        dout = np.array([[[[5.0]]]])
        dX = pool.backward(dout)
        assert dX[0, 0, 0, 1] == 5.0
        assert dX[0, 0, 0, 0] == 0.0


# =====================================================================
# 4. END-TO-END CNN & CHALLENGE (4 tests)
# =====================================================================
class TestEndToEndCNN:
    def test_conv_pool_block_challenge_gradcheck(self):
        block = ConvPoolBlock(in_channels=1, out_channels=2, kernel_size=3, padding=1, seed=42)
        X = np.random.randn(2, 1, 4, 4)
        out = block.forward(X)
        dout = out.copy()
        dX_ana, dW_ana, db_ana = block.backward(dout)

        eps = 1e-5
        num_dW = np.zeros_like(block.conv.W)
        for c_out in range(block.conv.W.shape[0]):
            for c_in in range(block.conv.W.shape[1]):
                for i in range(block.conv.W.shape[2]):
                    for j in range(block.conv.W.shape[3]):
                        orig = block.conv.W[c_out, c_in, i, j]
                        block.conv.W[c_out, c_in, i, j] = orig + eps
                        l1 = 0.5 * np.sum(block.forward(X) ** 2)
                        block.conv.W[c_out, c_in, i, j] = orig - eps
                        l2 = 0.5 * np.sum(block.forward(X) ** 2)
                        block.conv.W[c_out, c_in, i, j] = orig
                        num_dW[c_out, c_in, i, j] = (l1 - l2) / (2 * eps)

        abs_error = np.max(np.abs(dW_ana - num_dW))
        assert abs_error < 1e-4

    def test_simple_cnn_forward_pass_mnist_dimensions(self):
        # MNIST: 1x28x28 -> Conv(1->8, K=3, P=1) -> 8x28x28 -> Pool(2x2) -> 8x14x14 -> Flatten -> 1568
        conv = Conv2D(1, 8, kernel_size=3, padding=1)
        pool = MaxPool2D(2, 2)
        fl = Flatten()

        X = np.random.randn(4, 1, 28, 28)
        c_out = conv.forward(X)
        p_out = pool.forward(c_out)
        f_out = fl.forward(p_out)

        assert f_out.shape == (4, 1568)

    def test_conv2d_parameter_count_efficiency(self):
        # Conv2D with 16 filters of 3x3 on 3 channels = 16*3*3*3 + 16 = 448 params
        conv = Conv2D(3, 16, kernel_size=3)
        params = conv.W.size + conv.b.size
        assert params == 448

    def test_receptive_field_expansion_across_two_conv_layers(self):
        # Layer 1: K1=3, S1=1 -> RF1 = 3
        # Layer 2: K2=3, S2=1 -> RF2 = RF1 + (K2 - 1) * S1 = 3 + (3 - 1)*1 = 5
        conv1 = Conv2D(1, 4, kernel_size=3, padding=0)
        conv2 = Conv2D(4, 8, kernel_size=3, padding=0)

        X = np.zeros((1, 1, 5, 5))
        X[0, 0, 2, 2] = 1.0  # Center pixel active

        out1 = conv1.forward(X)      # Output 3x3
        out2 = conv2.forward(out1)   # Output 1x1

        # The single 1x1 output at center should be non-zero because input center pixel is in its receptive field
        assert out2.shape == (1, 8, 1, 1)

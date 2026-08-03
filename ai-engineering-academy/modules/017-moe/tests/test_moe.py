"""
AI ENGINEERING ACADEMY -- MODULE 017 TEST SUITE
Comprehensive Pytest Suite for Mixture-of-Experts (MoE) Architecture (16 Tests)
"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_mod17_dir = os.path.dirname(_dir)

_spec = importlib.util.spec_from_file_location("impl_mod17", os.path.join(_mod17_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ExpertFFN  = _mod.ExpertFFN
TopKRouter = _mod.TopKRouter
MoELayer   = _mod.MoELayer

_spec_ch = importlib.util.spec_from_file_location("ch_mod17", os.path.join(_mod17_dir, "07-challenge-solution.py"))
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)
MoETransformerBlock = _mod_ch.MoETransformerBlock
verify_moe_transformer_block = _mod_ch.verify_moe_transformer_block


# ===================================================================
# 1. EXPERT FFN (4 tests)
# ===================================================================
class TestExpertFFN:
    def test_expert_output_shape(self):
        exp = ExpertFFN(d_model=32, d_ff=64, seed=42)
        x = np.random.randn(10, 32)
        out = exp.forward(x)
        assert out.shape == (10, 32)

    def test_expert_no_nans(self):
        exp = ExpertFFN(d_model=32, d_ff=64, seed=42)
        x = np.random.randn(5, 32) * 10.0
        out = exp.forward(x)
        assert not np.isnan(out).any()

    def test_expert_zero_input(self):
        exp = ExpertFFN(d_model=16, d_ff=32, seed=42)
        x = np.zeros((4, 16))
        out = exp.forward(x)
        assert out.shape == (4, 16)

    def test_expert_single_token(self):
        exp = ExpertFFN(d_model=16, d_ff=32, seed=42)
        x = np.random.randn(1, 16)
        out = exp.forward(x)
        assert out.shape == (1, 16)


# ===================================================================
# 2. TOP-K ROUTER & AUX LOSS (4 tests)
# ===================================================================
class TestTopKRouter:
    def test_router_output_shapes(self):
        router = TopKRouter(d_model=32, num_experts=8, top_k=2, seed=42)
        x = np.random.randn(2, 5, 32)
        indices, weights, aux_loss = router.forward(x)
        assert indices.shape == (10, 2)  # 2*5 = 10 tokens, top_k=2
        assert weights.shape == (10, 2)
        assert isinstance(aux_loss, (float, np.floating))

    def test_router_weights_sum_to_one(self):
        router = TopKRouter(d_model=32, num_experts=4, top_k=2, seed=42)
        x = np.random.randn(2, 6, 32)
        _, weights, _ = router.forward(x)
        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_router_indices_valid_range(self):
        router = TopKRouter(d_model=32, num_experts=8, top_k=2, seed=42)
        x = np.random.randn(3, 4, 32)
        indices, _, _ = router.forward(x)
        assert (indices >= 0).all() and (indices < 8).all()

    def test_aux_loss_positive(self):
        router = TopKRouter(d_model=32, num_experts=8, top_k=2, seed=42)
        x = np.random.randn(4, 10, 32)
        _, _, aux_loss = router.forward(x)
        assert aux_loss > 0.0


# ===================================================================
# 3. MOE LAYER (4 tests)
# ===================================================================
class TestMoELayer:
    def test_moe_output_shape(self):
        moe = MoELayer(d_model=32, d_ff=64, num_experts=4, top_k=2, seed=42)
        x = np.random.randn(2, 6, 32)
        out, aux_loss = moe.forward(x)
        assert out.shape == (2, 6, 32)
        assert aux_loss > 0.0

    def test_moe_no_nans(self):
        moe = MoELayer(d_model=32, d_ff=64, num_experts=8, top_k=2, seed=42)
        x = np.random.randn(2, 8, 32) * 5.0
        out, _ = moe.forward(x)
        assert not np.isnan(out).any()

    def test_moe_top1_routing(self):
        moe = MoELayer(d_model=32, d_ff=64, num_experts=4, top_k=1, seed=42)
        x = np.random.randn(2, 4, 32)
        out, _ = moe.forward(x)
        assert out.shape == (2, 4, 32)

    def test_moe_different_input_produces_different_output(self):
        moe = MoELayer(d_model=32, d_ff=64, num_experts=4, top_k=2, seed=42)
        x1 = np.random.randn(1, 4, 32)
        x2 = np.random.randn(1, 4, 32)
        out1, _ = moe.forward(x1)
        out2, _ = moe.forward(x2)
        assert not np.allclose(out1, out2)


# ===================================================================
# 4. MOE TRANSFORMER BLOCK & CHALLENGE (4 tests)
# ===================================================================
class TestMoETransformerBlock:
    def test_block_output_shape(self):
        block = MoETransformerBlock(d_model=32, d_ff=64, num_experts=4, top_k=2, seed=42)
        X = np.random.randn(2, 6, 32)
        out, aux_loss = block.forward(X)
        assert out.shape == (2, 6, 32)
        assert aux_loss > 0.0

    def test_block_residual_connection_active(self):
        block = MoETransformerBlock(d_model=32, d_ff=64, num_experts=4, top_k=2, seed=42)
        X = np.random.randn(1, 4, 32)
        out, _ = block.forward(X)
        assert not np.allclose(out, X)

    def test_block_no_nans(self):
        block = MoETransformerBlock(d_model=32, d_ff=64, num_experts=4, top_k=2, seed=42)
        X = np.random.randn(2, 6, 32) * 5.0
        out, _ = block.forward(X)
        assert not np.isnan(out).any()

    def test_challenge_verification_runs(self):
        verify_moe_transformer_block()

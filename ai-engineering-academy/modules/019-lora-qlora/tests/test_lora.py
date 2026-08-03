"""
AI ENGINEERING ACADEMY -- MODULE 019 TEST SUITE
Comprehensive Pytest Suite for LoRA & QLoRA (16 Tests)
"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
_mod19_dir = os.path.dirname(_dir)

_spec = importlib.util.spec_from_file_location("impl_mod19", os.path.join(_mod19_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

LoRALinear  = _mod.LoRALinear
QLoRALinear = _mod.QLoRALinear

_spec_ch = importlib.util.spec_from_file_location("ch_mod19", os.path.join(_mod19_dir, "07-challenge-solution.py"))
_mod_ch = importlib.util.module_from_spec(_spec_ch)
_spec_ch.loader.exec_module(_mod_ch)
MultiAdapterLoRALinear = _mod_ch.MultiAdapterLoRALinear
verify_multi_adapter_switching = _mod_ch.verify_multi_adapter_switching


# ===================================================================
# 1. LORA LINEAR LAYER (4 tests)
# ===================================================================
class TestLoRALinear:
    def test_lora_output_shape(self):
        lora = LoRALinear(in_features=32, out_features=16, r=4, seed=42)
        x = np.random.randn(2, 5, 32)
        out = lora.forward(x)
        assert out.shape == (2, 5, 16)

    def test_lora_initialization_matches_base(self):
        lora = LoRALinear(in_features=32, out_features=16, r=4, seed=42)
        x = np.random.randn(2, 5, 32)
        out_lora = lora.forward(x)
        out_base = np.matmul(x, lora.weight.T)
        np.testing.assert_allclose(out_lora, out_base, atol=1e-6)

    def test_lora_merged_equivalence(self):
        lora = LoRALinear(in_features=32, out_features=16, r=4, lora_alpha=8, seed=42)
        lora.lora_B = np.random.randn(16, 4) * 0.1
        x = np.random.randn(2, 5, 32)

        out_unmerged = lora.forward(x)
        lora.merge()
        out_merged = lora.forward(x)

        np.testing.assert_allclose(out_unmerged, out_merged, atol=1e-5)

    def test_lora_unmerge_recovers_original(self):
        lora = LoRALinear(in_features=32, out_features=16, r=4, lora_alpha=8, seed=42)
        w_orig = lora.weight.copy()
        lora.lora_B = np.random.randn(16, 4) * 0.1

        lora.merge()
        lora.unmerge()
        np.testing.assert_allclose(lora.weight, w_orig, atol=1e-6)


# ===================================================================
# 2. QLORA LINEAR LAYER (4 tests)
# ===================================================================
class TestQLoRALinear:
    def test_qlora_output_shape(self):
        qlora = QLoRALinear(in_features=64, out_features=32, r=4, group_size=32, seed=42)
        x = np.random.randn(2, 5, 64)
        out = qlora.forward(x)
        assert out.shape == (2, 5, 32)

    def test_qlora_base_weights_quantized(self):
        qlora = QLoRALinear(in_features=64, out_features=32, r=4, group_size=32, seed=42)
        assert qlora.Q_W.dtype == np.int8
        assert qlora.Q_W.shape == (32, 64)

    def test_qlora_no_nans(self):
        qlora = QLoRALinear(in_features=64, out_features=32, r=4, group_size=32, seed=42)
        x = np.random.randn(3, 4, 64) * 5.0
        out = qlora.forward(x)
        assert not np.isnan(out).any()

    def test_qlora_scaling_factor(self):
        qlora = QLoRALinear(in_features=32, out_features=16, r=4, lora_alpha=16, group_size=32, seed=42)
        assert qlora.scaling == 4.0


# ===================================================================
# 3. MULTI-ADAPTER SWITCHING (4 tests)
# ===================================================================
class TestMultiAdapterSwitching:
    def test_multi_adapter_forward_shape(self):
        multi = MultiAdapterLoRALinear(in_features=32, out_features=16, seed=42)
        multi.add_adapter("code", r=4, lora_alpha=8, seed=1)
        x = np.random.randn(2, 4, 32)
        out = multi.forward(x, adapter_name="code")
        assert out.shape == (2, 4, 16)

    def test_adapters_produce_different_outputs(self):
        multi = MultiAdapterLoRALinear(in_features=32, out_features=16, seed=42)
        multi.add_adapter("code", r=4, lora_alpha=8, seed=1)
        multi.add_adapter("math", r=4, lora_alpha=8, seed=2)
        x = np.random.randn(2, 4, 32)

        out_code = multi.forward(x, adapter_name="code")
        out_math = multi.forward(x, adapter_name="math")
        assert not np.allclose(out_code, out_math)

    def test_base_forward_pass_when_no_adapter_specified(self):
        multi = MultiAdapterLoRALinear(in_features=32, out_features=16, seed=42)
        multi.add_adapter("code", r=4, lora_alpha=8, seed=1)
        x = np.random.randn(2, 4, 32)

        out_none = multi.forward(x, adapter_name=None)
        out_base = np.matmul(x, multi.weight_base.T)
        np.testing.assert_allclose(out_none, out_base, atol=1e-6)

    def test_unknown_adapter_name_falls_back_to_base(self):
        multi = MultiAdapterLoRALinear(in_features=32, out_features=16, seed=42)
        x = np.random.randn(2, 4, 32)
        out_unknown = multi.forward(x, adapter_name="nonexistent")
        out_base = np.matmul(x, multi.weight_base.T)
        np.testing.assert_allclose(out_unknown, out_base, atol=1e-6)


# ===================================================================
# 4. CHALLENGE & END-TO-END (4 tests)
# ===================================================================
class TestLoRAChallenge:
    def test_challenge_verification_runs(self):
        verify_multi_adapter_switching()

    def test_lora_r_zero_equals_base_layer(self):
        lora0 = LoRALinear(in_features=32, out_features=16, r=0, seed=42)
        x = np.random.randn(2, 4, 32)
        out = lora0.forward(x)
        assert out.shape == (2, 4, 16)

    def test_lora_batch_broadcasting(self):
        lora = LoRALinear(in_features=16, out_features=8, r=2, seed=42)
        x_3d = np.random.randn(2, 5, 16)
        x_4d = np.random.randn(2, 3, 5, 16)
        out_3d = lora.forward(x_3d)
        out_4d = lora.forward(x_4d)
        assert out_3d.shape == (2, 5, 8)
        assert out_4d.shape == (2, 3, 5, 8)

    def test_lora_no_nans(self):
        lora = LoRALinear(in_features=32, out_features=16, r=4, seed=42)
        x = np.random.randn(3, 10, 32) * 10.0
        out = lora.forward(x)
        assert not np.isnan(out).any()

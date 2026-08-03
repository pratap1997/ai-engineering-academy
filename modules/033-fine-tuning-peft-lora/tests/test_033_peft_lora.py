import sys
import os
import numpy as np
import importlib.util

def load_module():
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "04-implementation.py")
    spec = importlib.util.spec_from_file_location("module_033_impl", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module_033_impl"] = module
    spec.loader.exec_module(module)
    return module

module = load_module()
LinearLayer = module.LinearLayer
LoRALayer = module.LoRALayer
QuantizedWeights = module.QuantizedWeights
SFTTrainer = module.SFTTrainer

# CATEGORY 1: LoRA Matrix Math
def test_lora_dims():
    base = LinearLayer(128, 64)
    lora = LoRALayer(base, rank=8)
    assert lora.A.shape == (8, 128)
    assert lora.B.shape == (64, 8)

def test_lora_init_zero():
    base = LinearLayer(128, 64)
    lora = LoRALayer(base, rank=8)
    assert np.all(lora.B == 0)

def test_lora_alpha_scaling():
    base = LinearLayer(128, 64)
    lora = LoRALayer(base, rank=4, alpha=16)
    assert lora.scaling == 4.0

def test_lora_merge():
    base = LinearLayer(10, 5)
    lora = LoRALayer(base, rank=2, alpha=2)
    lora.B = np.ones((5, 2))
    merged = lora.merge()
    assert merged.shape == (5, 10)
    expected = base.weight + (lora.B @ lora.A) * lora.scaling
    np.testing.assert_array_almost_equal(merged, expected)

# CATEGORY 2: Forward/Backward Pass
def test_forward_initial_equivalence():
    base = LinearLayer(10, 5)
    lora = LoRALayer(base, rank=2)
    x = np.random.randn(3, 10)
    np.testing.assert_array_almost_equal(base.forward(x), lora.forward(x))

def test_forward_updated_b():
    base = LinearLayer(10, 5)
    lora = LoRALayer(base, rank=2, alpha=2)
    lora.B = np.ones((5, 2))
    x = np.random.randn(3, 10)
    assert not np.allclose(base.forward(x), lora.forward(x))

def test_batch_handling():
    base = LinearLayer(10, 5)
    lora = LoRALayer(base, rank=2)
    assert lora.forward(np.random.randn(1, 10)).shape == (1, 5)
    assert lora.forward(np.random.randn(4, 10)).shape == (4, 5)

def test_activation_addition():
    base = LinearLayer(10, 5)
    lora = LoRALayer(base, rank=2, alpha=2)
    lora.B = np.ones((5, 2))
    x = np.ones((1, 10))
    lora_component = (x @ lora.A.T @ lora.B.T) * lora.scaling
    np.testing.assert_array_almost_equal(lora.forward(x), base.forward(x) + lora_component)

# CATEGORY 3: Trainer & Data Formatting
def test_sft_data_format():
    trainer = SFTTrainer(None)
    res = trainer.format_sft_data("X", "Y")
    assert "Instruction:\nX" in res
    assert "Response:\nY" in res

def test_trainer_init():
    trainer = SFTTrainer("model", learning_rate=0.01)
    assert trainer.lr == 0.01
    assert trainer.model == "model"

def test_loss_computation():
    trainer = SFTTrainer(None)
    loss = trainer.compute_loss(np.array([0.5, 0.2]), np.array([1.0, 0.0]))
    assert np.isclose(loss, 0.145)

def test_quantization_bits():
    qw8 = QuantizedWeights(np.random.randn(10, 10), bits=8)
    qw4 = QuantizedWeights(np.random.randn(10, 10), bits=4)
    assert qw8.bits == 8
    assert qw4.bits == 4

# CATEGORY 4: Parameter Efficiency Math
def test_param_count_base():
    base = LinearLayer(100, 50)
    assert base.weight.size + base.bias.size == 5050

def test_param_count_lora():
    base = LinearLayer(100, 50)
    lora = LoRALayer(base, rank=4)
    assert lora.A.size + lora.B.size == 600

def test_savings_ratio():
    assert ((4 * 100) + (50 * 4)) < (100 * 50)

def test_dequantize_reconstruction():
    weights = np.random.uniform(-1, 1, (10, 10))
    qw = QuantizedWeights(weights, bits=8)
    mse = np.mean((weights - qw.dequantize())**2)
    assert mse < 0.1

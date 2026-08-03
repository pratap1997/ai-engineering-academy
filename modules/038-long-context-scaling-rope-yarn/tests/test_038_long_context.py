import pytest
import numpy as np
import importlib.util
import sys
import os

def load_module():
    module_name = "module_038_impl"
    file_path = os.path.join(os.path.dirname(__file__), "..", "04-implementation.py")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

impl = load_module()

# 1. RoPE Matrix Rotation
def test_rope_matrix_shape():
    rope = impl.StandardRoPE(d_model=64)
    x = np.random.randn(10, 64)
    out = rope(x)
    assert out.shape == (10, 64)

def test_rope_zero_position():
    rope = impl.StandardRoPE(d_model=64)
    x = np.ones((1, 64))
    out = rope(x)
    np.testing.assert_allclose(out, x, rtol=1e-5)

def test_rope_preserves_norm():
    rope = impl.StandardRoPE(d_model=64)
    x = np.random.randn(10, 64)
    out = rope(x)
    np.testing.assert_allclose(np.linalg.norm(out, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5)

def test_get_rotary_matrix_frequencies():
    freqs = impl.get_rotary_matrix(10, 64)
    assert freqs.shape == (10, 32)
    np.testing.assert_allclose(freqs[0], np.zeros(32))

# 2. RoPE Scaling Algorithms
def test_linear_scaled_rope():
    std_rope = impl.StandardRoPE(d_model=64)
    lin_rope = impl.LinearScaledRoPE(d_model=64, scale=2.0)
    
    x_pos2 = np.ones((3, 64))
    x_pos1 = np.ones((2, 64))
    
    out_std = std_rope(x_pos1)
    out_lin = lin_rope(x_pos2)
    
    np.testing.assert_allclose(out_lin[2], out_std[1], rtol=1e-5)

def test_ntk_aware_rope():
    ntk_rope = impl.NTKAwareRoPE(d_model=64, scale=2.0)
    x = np.ones((10, 64))
    out = ntk_rope(x)
    assert out.shape == (10, 64)

def test_yarn_rope():
    yarn_rope = impl.YaRNRoPE(d_model=64, scale=2.0)
    x = np.ones((10, 64))
    out = yarn_rope(x)
    assert out.shape == (10, 64)

def test_scaling_distinguishes_positions():
    lin_rope = impl.LinearScaledRoPE(d_model=64, scale=2.0)
    x = np.ones((2, 64))
    out = lin_rope(x)
    assert not np.allclose(out[0], out[1])

# 3. Sliding Window Masking
def test_swa_mask_shape():
    mask = impl.SlidingWindowAttentionMask(10, window_size=3)
    assert mask.shape == (10, 10)

def test_swa_mask_causal():
    mask = impl.SlidingWindowAttentionMask(10, window_size=3)
    assert np.all(mask[np.triu_indices(10, k=1)] == -np.inf)

def test_swa_mask_window():
    mask = impl.SlidingWindowAttentionMask(10, window_size=3)
    for i in range(4, 10):
        assert mask[i, i-4] == -np.inf
    for i in range(3, 10):
        assert mask[i, i-3] == 0.0

def test_swa_mask_diagonal():
    mask = impl.SlidingWindowAttentionMask(10, window_size=3)
    assert np.all(np.diag(mask) == 0.0)

# 4. Chunked Prefill Logic
def test_chunked_prefill_engine_init():
    engine = impl.ChunkedPrefillEngine(chunk_size=128)
    assert engine.chunk_size == 128
    assert len(engine.kv_cache) == 0

def test_chunked_prefill_exact_multiple():
    engine = impl.ChunkedPrefillEngine(chunk_size=10)
    seq = np.arange(30)
    chunks = engine.prefill(seq)
    assert len(chunks) == 3
    assert engine.get_cache_size() == 30

def test_chunked_prefill_remainder():
    engine = impl.ChunkedPrefillEngine(chunk_size=10)
    seq = np.arange(35)
    chunks = engine.prefill(seq)
    assert len(chunks) == 4
    assert len(chunks[-1]) == 5
    assert engine.get_cache_size() == 35

def test_chunked_prefill_smaller_than_chunk():
    engine = impl.ChunkedPrefillEngine(chunk_size=100)
    seq = np.arange(25)
    chunks = engine.prefill(seq)
    assert len(chunks) == 1
    assert len(chunks[0]) == 25
    assert engine.get_cache_size() == 25

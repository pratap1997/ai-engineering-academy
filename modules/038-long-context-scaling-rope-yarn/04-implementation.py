import numpy as np
from typing import Tuple, List, Optional

def get_rotary_matrix(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """Standard RoPE matrix frequencies."""
    assert d_model % 2 == 0
    theta = 1.0 / (base ** (np.arange(0, d_model, 2)[: (d_model // 2)].astype(float) / d_model))
    m = np.arange(seq_len)
    freqs = np.outer(m, theta)
    return freqs

def apply_rotary_emb(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Applies RoPE to a tensor of shape (seq_len, d_model)."""
    seq_len, d_model = x.shape
    x_reshaped = x.reshape(seq_len, d_model // 2, 2)
    x1, x2 = x_reshaped[:, :, 0], x_reshaped[:, :, 1]
    
    sin_freqs = np.sin(freqs)
    cos_freqs = np.cos(freqs)
    
    out1 = x1 * cos_freqs - x2 * sin_freqs
    out2 = x2 * cos_freqs + x1 * sin_freqs
    
    return np.stack([out1, out2], axis=-1).reshape(seq_len, d_model)

class StandardRoPE:
    def __init__(self, d_model: int, max_seq_len: int = 2048, base: float = 10000.0):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[0]
        freqs = get_rotary_matrix(seq_len, self.d_model, self.base)
        return apply_rotary_emb(x, freqs)

class LinearScaledRoPE:
    def __init__(self, d_model: int, max_seq_len: int = 2048, scale: float = 2.0, base: float = 10000.0):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.scale = scale
        self.base = base
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[0]
        theta = 1.0 / (self.base ** (np.arange(0, self.d_model, 2).astype(float) / self.d_model))
        m = np.arange(seq_len) / self.scale
        freqs = np.outer(m, theta)
        return apply_rotary_emb(x, freqs)

class NTKAwareRoPE:
    def __init__(self, d_model: int, max_seq_len: int = 2048, scale: float = 2.0, base: float = 10000.0):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.scale = scale
        self.base = base
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[0]
        new_base = self.base * (self.scale ** (self.d_model / (self.d_model - 2)))
        theta = 1.0 / (new_base ** (np.arange(0, self.d_model, 2).astype(float) / self.d_model))
        m = np.arange(seq_len)
        freqs = np.outer(m, theta)
        return apply_rotary_emb(x, freqs)

class YaRNRoPE:
    def __init__(self, d_model: int, scale: float = 2.0, base: float = 10000.0, beta_fast: float = 32.0, beta_slow: float = 1.0):
        self.d_model = d_model
        self.scale = scale
        self.base = base
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[0]
        i = np.arange(0, self.d_model, 2).astype(float)
        theta_orig = self.base ** (i / self.d_model)
        
        m = (self.d_model / (2 * np.log(self.base))) * np.log(theta_orig)
        
        gamma = np.zeros_like(i)
        idx_slow = m < self.beta_slow
        idx_fast = m > self.beta_fast
        idx_mid = ~(idx_slow | idx_fast)
        
        gamma[idx_fast] = 0.0
        gamma[idx_slow] = 1.0
        gamma[idx_mid] = (m[idx_mid] - self.beta_fast) / (self.beta_slow - self.beta_fast)
        
        theta_scaled = theta_orig * ((self.scale) ** gamma)
        theta = 1.0 / theta_scaled
        
        positions = np.arange(seq_len)
        freqs = np.outer(positions, theta)
        
        t = np.sqrt(1 + 0.1 * np.log(self.scale))
        
        return apply_rotary_emb(x, freqs) * t

def SlidingWindowAttentionMask(seq_len: int, window_size: int) -> np.ndarray:
    """Creates a sliding window attention mask."""
    indices = np.arange(seq_len)
    dist = indices[:, None] - indices[None, :]
    mask = (dist >= 0) & (dist <= window_size)
    
    out = np.full((seq_len, seq_len), -np.inf)
    out[mask] = 0.0
    return out

class ChunkedPrefillEngine:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self.kv_cache = []
        
    def prefill(self, sequence: np.ndarray) -> List[np.ndarray]:
        """Simulates processing a long sequence in chunks."""
        seq_len = sequence.shape[0]
        chunks = []
        for i in range(0, seq_len, self.chunk_size):
            chunk = sequence[i:i + self.chunk_size]
            chunks.append(chunk)
            self.kv_cache.append(chunk)
        return chunks
        
    def get_cache_size(self):
        return sum(c.shape[0] for c in self.kv_cache)

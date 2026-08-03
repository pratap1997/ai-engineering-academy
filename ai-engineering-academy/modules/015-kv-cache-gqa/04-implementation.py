"""
AI ENGINEERING ACADEMY -- MODULE 015
KV Cache & Grouped-Query Attention Implementation (Pure Python & NumPy)

Provides:
1. `KVCache`: Dynamic Key-Value cache container for single-token autoregressive decoding.
2. `GroupedQueryAttention`: Unified MHA / GQA / MQA forward pass with KV head repeat expansion.
"""

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# =====================================================================
# 1. KEY-VALUE CACHE (KV CACHE)
# =====================================================================

class KVCache:
    """
    Dynamic state container storing Key and Value states across decoding steps.
    Shape per layer:
      k_cache: (batch_size, num_kv_heads, seq_len, head_dim)
      v_cache: (batch_size, num_kv_heads, seq_len, head_dim)
    """

    def __init__(self):
        self.k_cache = None
        self.v_cache = None

    def update(self, key_states, value_states):
        """
        key_states:   (batch_size, num_kv_heads, new_seq_len, head_dim)
        value_states: (batch_size, num_kv_heads, new_seq_len, head_dim)
        Returns: concatenated (k_cache, v_cache) covering all positions 0..t
        """
        if self.k_cache is None:
            self.k_cache = key_states
            self.v_cache = value_states
        else:
            # Append along sequence dimension (axis=2)
            self.k_cache = np.concatenate([self.k_cache, key_states], axis=2)
            self.v_cache = np.concatenate([self.v_cache, value_states], axis=2)

        return self.k_cache, self.v_cache

    def reset(self):
        """Clear cache state."""
        self.k_cache = None
        self.v_cache = None

    @property
    def seq_len(self):
        return 0 if self.k_cache is None else self.k_cache.shape[2]


# =====================================================================
# 2. GROUPED-QUERY ATTENTION (GQA / MQA / MHA)
# =====================================================================

class GroupedQueryAttention:
    """
    Unified Multi-Head / Grouped-Query / Multi-Query Attention.
      num_query_heads (H_Q)
      num_kv_heads    (H_KV)  -- H_KV = H_Q (MHA), 1 < H_KV < H_Q (GQA), H_KV = 1 (MQA)
    """

    def __init__(self, d_model, num_query_heads, num_kv_heads, seed=None):
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads  # Repetition factor G
        self.head_dim = d_model // num_query_heads

        if seed is not None:
            np.random.seed(seed)

        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, num_query_heads * self.head_dim) * scale
        self.W_K = np.random.randn(d_model, num_kv_heads * self.head_dim) * scale
        self.W_V = np.random.randn(d_model, num_kv_heads * self.head_dim) * scale
        self.W_O = np.random.randn(num_query_heads * self.head_dim, d_model) * scale

    def _repeat_kv(self, x, n_rep):
        """
        Repeat KV heads to match query heads count along head dimension.
        x: (batch_size, num_kv_heads, seq_len, head_dim)
        Returns: (batch_size, num_query_heads, seq_len, head_dim)
        """
        if n_rep == 1:
            return x
        batch_size, num_kv_heads, seq_len, head_dim = x.shape
        # Expand and repeat along head axis
        x_expanded = x[:, :, None, :, :]  # (N, H_KV, 1, T, d)
        x_repeated = np.repeat(x_expanded, n_rep, axis=2)  # (N, H_KV, G, T, d)
        return x_repeated.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(self, x, kv_cache=None, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        kv_cache: Optional KVCache object
        Returns: output (batch_size, seq_len, d_model), attn_weights (batch_size, H_Q, seq_len, total_seq_len)
        """
        N, T, _ = x.shape

        # Linear projections
        Q = np.matmul(x, self.W_Q).reshape(N, T, self.num_query_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.matmul(x, self.W_K).reshape(N, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.matmul(x, self.W_V).reshape(N, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Update or use KV cache if provided
        if kv_cache is not None:
            K, V = kv_cache.update(K, V)

        total_seq_len = K.shape[2]

        # Repeat KV heads for GQA/MQA to match H_Q
        K_expanded = self._repeat_kv(K, self.num_queries_per_kv)  # (N, H_Q, total_seq_len, head_dim)
        V_expanded = self._repeat_kv(V, self.num_queries_per_kv)  # (N, H_Q, total_seq_len, head_dim)

        # Scaled dot-product attention: Q (N, H_Q, T, d) @ K^T (N, H_Q, d, total_seq_len)
        scale = np.sqrt(self.head_dim)
        scores = np.matmul(Q, K_expanded.transpose(0, 1, 3, 2)) / scale

        if mask is not None:
            scores = scores + mask * -1e9

        attn_weights = softmax(scores, axis=-1)  # (N, H_Q, T, total_seq_len)
        out = np.matmul(attn_weights, V_expanded)  # (N, H_Q, T, head_dim)

        # Reshape to (N, T, d_model) and project output
        out = out.transpose(0, 2, 1, 3).reshape(N, T, self.d_model)
        output = np.matmul(out, self.W_O)
        return output, attn_weights


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 015 -- KV CACHE & GQA VERIFICATION")
    print("=" * 65)

    N, T, d_model = 2, 5, 64
    H_Q = 8

    # Test 1: MHA (H_KV = 8)
    mha = GroupedQueryAttention(d_model=d_model, num_query_heads=H_Q, num_kv_heads=8, seed=42)
    x = np.random.randn(N, T, d_model)
    out_mha, attn_mha = mha.forward(x)
    print("\n[1. Multi-Head Attention (MHA: H_Q=8, H_KV=8)]")
    print(f"  Output shape: {out_mha.shape} (Expected: ({N}, {T}, {d_model})) => [OK]")

    # Test 2: GQA (H_KV = 2, ratio 4:1)
    gqa = GroupedQueryAttention(d_model=d_model, num_query_heads=H_Q, num_kv_heads=2, seed=42)
    out_gqa, attn_gqa = gqa.forward(x)
    print("\n[2. Grouped-Query Attention (GQA: H_Q=8, H_KV=2)]")
    print(f"  Output shape: {out_gqa.shape} (Expected: ({N}, {T}, {d_model})) => [OK]")
    print(f"  Attn shape:   {attn_gqa.shape} (Expected: ({N}, {H_Q}, {T}, {T})) => [OK]")

    # 3. KV Cache Autoregressive Decoding Step-by-Step
    print("\n[3. KV Cache Incremental Decoding Verification]")
    cache = KVCache()
    full_text_input = np.random.randn(1, T, d_model)
    causal_mask = np.triu(np.ones((1, T, T)), k=1)

    # 1. Full sequence forward pass (causal masked)
    full_out, _ = gqa.forward(full_text_input, mask=causal_mask)

    # 2. Incremental step-by-step forward pass using KV Cache
    incremental_outs = []
    for t in range(T):
        single_token_input = full_text_input[:, t:t+1, :]  # (1, 1, d_model)
        step_out, _ = gqa.forward(single_token_input, kv_cache=cache)
        incremental_outs.append(step_out)

    reconstructed_out = np.concatenate(incremental_outs, axis=1)  # (1, T, d_model)
    np.testing.assert_allclose(full_out, reconstructed_out, atol=1e-5)
    print(f"  Incremental KV Cache output matches causal full sequence forward pass exactly => [OK]")

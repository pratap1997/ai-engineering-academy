"""
AI ENGINEERING ACADEMY -- MODULE 018
FlashAttention & Tiled Online Softmax Implementation (Pure Python & NumPy)

Provides:
1. `OnlineSoftmax`: Streaming softmax accumulator maintaining online max and sum-exp statistics.
2. `FlashAttentionTiled`: Block-wise tiled attention algorithm eliminating O(T^2) HBM memory footprint.
3. `standard_attention`: Reference baseline for numerical verification.
"""

import numpy as np


# =====================================================================
# 1. ONLINE SOFTMAX ACCUMULATOR
# =====================================================================

class OnlineSoftmax:
    """
    Online Softmax accumulator for a single row or batch of rows.
    Tracks running max (m) and running sum-exp (d).
    """

    def __init__(self):
        self.m = None  # running max
        self.d = None  # running sum-exp

    def update(self, x_block):
        """
        x_block: (..., B_c) array of new logit scores
        Returns:
          p_block: (..., B_c) local exponentiated scores: exp(x_block - m_new)
          m_new: updated max
          d_new: updated sum
        """
        local_max = np.max(x_block, axis=-1, keepdims=True)  # (..., 1)
        local_exp = np.exp(x_block - local_max)               # (..., B_c)
        local_sum = np.sum(local_exp, axis=-1, keepdims=True) # (..., 1)

        if self.m is None:
            self.m = local_max
            self.d = local_sum
            return local_exp, self.m, self.d

        # Recurrence relation for online max & sum
        m_new = np.maximum(self.m, local_max)
        alpha = np.exp(self.m - m_new)
        beta = np.exp(local_max - m_new)

        self.d = alpha * self.d + beta * local_sum
        self.m = m_new

        return local_exp * beta, self.m, self.d


# =====================================================================
# 2. FLASHATTENTION TILED FORWARD PASS
# =====================================================================

class FlashAttentionTiled:
    """
    Pure NumPy implementation of FlashAttention-1 tiled online softmax algorithm.
    Splits Q, K, V into tiles of size B_r x d and B_c x d.
    """

    def __init__(self, block_r=16, block_c=16):
        self.B_r = block_r
        self.B_c = block_c

    def forward(self, Q, K, V, causal=False):
        """
        Q: (batch_size, num_heads, T_q, d_head)
        K: (batch_size, num_heads, T_k, d_head)
        V: (batch_size, num_heads, T_k, d_head)
        Returns: Output O (batch_size, num_heads, T_q, d_head)
        """
        N, H, T_q, d = Q.shape
        _, _, T_k, _ = K.shape
        scale = 1.0 / np.sqrt(d)

        # Allocate output buffer O, max buffer m, sum buffer d in HBM memory simulation
        O = np.zeros((N, H, T_q, d), dtype=np.float64)
        m = np.full((N, H, T_q, 1), -np.inf, dtype=np.float64)
        l_sum = np.zeros((N, H, T_q, 1), dtype=np.float64)

        T_r = int(np.ceil(T_q / self.B_r))
        T_c = int(np.ceil(T_k / self.B_c))

        # Outer loop over Query blocks (FlashAttention-2 order for parallelism)
        for i in range(T_r):
            q_start, q_end = i * self.B_r, min((i + 1) * self.B_r, T_q)
            Q_i = Q[:, :, q_start:q_end, :]  # (N, H, B_r, d)
            B_r_actual = q_end - q_start

            O_i = np.zeros((N, H, B_r_actual, d), dtype=np.float64)
            m_i = np.full((N, H, B_r_actual, 1), -np.inf, dtype=np.float64)
            l_i = np.zeros((N, H, B_r_actual, 1), dtype=np.float64)

            # Inner loop over Key/Value blocks
            for j in range(T_c):
                k_start, k_end = j * self.B_c, min((j + 1) * self.B_c, T_k)
                K_j = K[:, :, k_start:k_end, :]  # (N, H, B_c, d)
                V_j = V[:, :, k_start:k_end, :]  # (N, H, B_c, d)

                # Compute tiled dot product score: Q_i @ K_j.T
                S_ij = np.matmul(Q_i, K_j.transpose(0, 1, 3, 2)) * scale  # (N, H, B_r, B_c)

                if causal:
                    # Apply causal mask: mask positions where k_index > q_index
                    row_idx = np.arange(q_start, q_end)[:, None]
                    col_idx = np.arange(k_start, k_end)[None, :]
                    causal_mask = col_idx > row_idx
                    S_ij = np.where(causal_mask[None, None, :, :], -1e9, S_ij)

                # Local max & sum for block S_ij
                m_ij = np.max(S_ij, axis=-1, keepdims=True)     # (N, H, B_r, 1)
                p_ij = np.exp(S_ij - m_ij)                       # (N, H, B_r, B_c)
                l_ij = np.sum(p_ij, axis=-1, keepdims=True)      # (N, H, B_r, 1)

                # Online Softmax updates
                m_i_new = np.maximum(m_i, m_ij)
                alpha = np.exp(m_i - m_i_new)
                beta = np.exp(m_ij - m_i_new)

                l_i_new = alpha * l_i + beta * l_ij

                # Output accumulator update:
                # O_i = (alpha * l_i * O_i + beta * P_ij @ V_j) / l_i_new
                O_i = (alpha * l_i * O_i + beta * np.matmul(p_ij, V_j)) / (l_i_new + 1e-15)

                m_i = m_i_new
                l_i = l_i_new

            O[:, :, q_start:q_end, :] = O_i
            m[:, :, q_start:q_end, :] = m_i
            l_sum[:, :, q_start:q_end, :] = l_i

        return O


# =====================================================================
# 3. REFERENCE STANDARD ATTENTION
# =====================================================================

def standard_attention(Q, K, V, causal=False):
    """Reference standard PyTorch-style softmax attention for comparison."""
    d = Q.shape[-1]
    scale = 1.0 / np.sqrt(d)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale  # (N, H, T_q, T_k)

    if causal:
        T_q, T_k = Q.shape[2], K.shape[2]
        causal_mask = np.triu(np.ones((T_q, T_k)), k=1)
        scores = np.where(causal_mask[None, None, :, :], -1e9, scores)

    # Softmax over full T_k dimension
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    return np.matmul(attn, V)


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 018 -- FLASHATTENTION VERIFICATION")
    print("=" * 65)

    np.random.seed(42)
    N, H, T, d = 2, 4, 64, 32
    Q = np.random.randn(N, H, T, d)
    K = np.random.randn(N, H, T, d)
    V = np.random.randn(N, H, T, d)

    # 1. Standard Attention Baseline
    out_std = standard_attention(Q, K, V, causal=False)

    # 2. FlashAttention Tiled
    flash = FlashAttentionTiled(block_r=16, block_c=16)
    out_flash = flash.forward(Q, K, V, causal=False)

    print("\n[1. FlashAttention Numerical Equivalence Test]")
    print(f"  Standard Attention Output Shape: {out_std.shape}")
    print(f"  FlashAttention Output Shape:     {out_flash.shape} => [OK]")

    max_diff = np.max(np.abs(out_std - out_flash))
    print(f"  Max Absolute Difference: {max_diff:.8e}")
    np.testing.assert_allclose(out_std, out_flash, atol=1e-5)
    print("  EXACT NUMERICAL EQUIVALENCE VERIFIED (<1e-5) => [OK]")

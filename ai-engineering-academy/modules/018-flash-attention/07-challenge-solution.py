"""
AI ENGINEERING ACADEMY -- MODULE 018 ENGINEERING CHALLENGE SOLUTION
Causal FlashAttention with Upper-Triangular Tile Skipping
"""

import os
import importlib.util
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("impl_mod18", os.path.join(_dir, "04-implementation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FlashAttentionTiled = _mod.FlashAttentionTiled
standard_attention  = _mod.standard_attention


class CausalFlashAttention:
    """
    FlashAttention with causal masking & upper-triangular block tile skipping.
    """

    def __init__(self, block_r=16, block_c=16):
        self.B_r = block_r
        self.B_c = block_c

    def forward(self, Q, K, V):
        N, H, T_q, d = Q.shape
        _, _, T_k, _ = K.shape
        scale = 1.0 / np.sqrt(d)

        O = np.zeros((N, H, T_q, d), dtype=np.float64)

        T_r = int(np.ceil(T_q / self.B_r))
        T_c = int(np.ceil(T_k / self.B_c))

        tiles_processed = 0
        tiles_skipped = 0

        for i in range(T_r):
            q_start, q_end = i * self.B_r, min((i + 1) * self.B_r, T_q)
            Q_i = Q[:, :, q_start:q_end, :]
            B_r_actual = q_end - q_start

            O_i = np.zeros((N, H, B_r_actual, d), dtype=np.float64)
            m_i = np.full((N, H, B_r_actual, 1), -np.inf, dtype=np.float64)
            l_i = np.zeros((N, H, B_r_actual, 1), dtype=np.float64)

            for j in range(T_c):
                k_start, k_end = j * self.B_c, min((j + 1) * self.B_c, T_k)

                # Skip upper triangular blocks where k_start > q_end - 1
                if k_start >= q_end:
                    tiles_skipped += 1
                    continue

                tiles_processed += 1
                K_j = K[:, :, k_start:k_end, :]
                V_j = V[:, :, k_start:k_end, :]

                S_ij = np.matmul(Q_i, K_j.transpose(0, 1, 3, 2)) * scale

                # Apply causal mask inside tile
                row_idx = np.arange(q_start, q_end)[:, None]
                col_idx = np.arange(k_start, k_end)[None, :]
                causal_mask = col_idx > row_idx
                S_ij = np.where(causal_mask[None, None, :, :], -1e9, S_ij)

                m_ij = np.max(S_ij, axis=-1, keepdims=True)
                p_ij = np.exp(S_ij - m_ij)
                l_ij = np.sum(p_ij, axis=-1, keepdims=True)

                m_i_new = np.maximum(m_i, m_ij)
                alpha = np.exp(m_i - m_i_new)
                beta = np.exp(m_ij - m_i_new)

                l_i_new = alpha * l_i + beta * l_ij

                O_i = (alpha * l_i * O_i + beta * np.matmul(p_ij, V_j)) / (l_i_new + 1e-15)

                m_i = m_i_new
                l_i = l_i_new

            O[:, :, q_start:q_end, :] = O_i

        return O, tiles_processed, tiles_skipped


def verify_causal_flash_attention():
    print("=" * 65)
    print("MODULE 018 CHALLENGE: CAUSAL FLASHATTENTION WITH TILE SKIPPING")
    print("=" * 65)

    np.random.seed(42)
    N, H, T, d = 1, 2, 64, 16
    Q = np.random.randn(N, H, T, d)
    K = np.random.randn(N, H, T, d)
    V = np.random.randn(N, H, T, d)

    out_std = standard_attention(Q, K, V, causal=True)

    causal_flash = CausalFlashAttention(block_r=16, block_c=16)
    out_flash, processed, skipped = causal_flash.forward(Q, K, V)

    print(f"Total Block Tiles:    {processed + skipped}")
    print(f"Processed Tiles:      {processed}")
    print(f"Skipped Tiles:        {skipped} (~{(skipped / (processed + skipped)) * 100:.1f}% saved)")

    max_diff = np.max(np.abs(out_std - out_flash))
    print(f"Max Absolute Diff:    {max_diff:.8e}")
    np.testing.assert_allclose(out_std, out_flash, atol=1e-5)

    print("\nCausal FlashAttention Numerical Verification Passed [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_causal_flash_attention()

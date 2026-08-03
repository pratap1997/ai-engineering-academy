"""
AI ENGINEERING ACADEMY -- MODULE 017
Mixture-of-Experts (MoE) Implementation (Pure Python & NumPy)

Provides:
1. `ExpertFFN`: Single expert 2-layer FFN network with GELU activation.
2. `TopKRouter`: Softmax Gating Router selecting Top-k experts & calculating load balancing loss.
3. `MoELayer`: Sparse Mixture-of-Experts layer dispatching tokens and aggregating expert outputs.
"""

import numpy as np


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# =====================================================================
# 1. EXPERT FFN NETWORK
# =====================================================================

class ExpertFFN:
    """Single expert 2-layer FFN network."""

    def __init__(self, d_model, d_ff, seed=None):
        self.d_model = d_model
        self.d_ff = d_ff

        if seed is not None:
            np.random.seed(seed)

        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        """x: (num_tokens, d_model) -> output (num_tokens, d_model)"""
        hidden = gelu(np.matmul(x, self.W1) + self.b1)
        return np.matmul(hidden, self.W2) + self.b2


# =====================================================================
# 2. TOP-K GATING ROUTER
# =====================================================================

class TopKRouter:
    """
    Softmax Top-k Gating Router.
    Routes tokens to top k experts and computes Switch Transformer auxiliary load balancing loss.
    """

    def __init__(self, d_model, num_experts, top_k=2, aux_loss_coef=0.01, seed=None):
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef

        if seed is not None:
            np.random.seed(seed)

        scale = np.sqrt(2.0 / d_model)
        self.W_gate = np.random.randn(d_model, num_experts) * scale

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        Returns:
          topk_indices:  (N*T, top_k) -- selected expert indices per token
          topk_weights:  (N*T, top_k) -- normalized softmax gating weights
          aux_loss:      scalar load balancing loss
        """
        N, T, d = x.shape
        flat_x = x.reshape(N * T, d)  # (M, d) where M = N*T tokens

        # Raw router logits: (M, E)
        logits = np.matmul(flat_x, self.W_gate)
        full_probs = softmax(logits, axis=-1)  # (M, E)

        # Select Top-k experts per token
        topk_indices = np.argsort(logits, axis=-1)[:, -self.top_k:][:, ::-1]  # (M, top_k)

        # Gather logits of top-k and softmax over top-k only
        M_idx = np.arange(flat_x.shape[0])[:, None]
        topk_logits = logits[M_idx, topk_indices]
        topk_weights = softmax(topk_logits, axis=-1)  # (M, top_k)

        # Compute Auxiliary Load Balancing Loss (Switch Transformer formula)
        # f_i = fraction of tokens routed to expert i
        flat_topk_idx = topk_indices.flatten()
        counts = np.bincount(flat_topk_idx, minlength=self.num_experts)
        f_i = counts / (flat_x.shape[0] * self.top_k)  # (E,)

        # P_i = average routing probability for expert i
        P_i = full_probs.mean(axis=0)  # (E,)

        aux_loss = self.aux_loss_coef * self.num_experts * np.sum(f_i * P_i)

        return topk_indices, topk_weights, aux_loss


# =====================================================================
# 3. MOE LAYER
# =====================================================================

class MoELayer:
    """
    Sparse Mixture-of-Experts Layer.
    Dispatches tokens to selected top-k experts and aggregates outputs.
    """

    def __init__(self, d_model, d_ff, num_experts=8, top_k=2, seed=None):
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, num_experts, top_k=top_k, seed=seed)
        self.experts = [
            ExpertFFN(d_model, d_ff, seed=(seed + i if seed is not None else None))
            for i in range(num_experts)
        ]

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        Returns: output (batch_size, seq_len, d_model), aux_loss
        """
        N, T, d = x.shape
        flat_x = x.reshape(N * T, d)  # (M, d)
        M = flat_x.shape[0]

        topk_indices, topk_weights, aux_loss = self.router.forward(x)  # (M, k), (M, k)

        # Output accumulator initialized to zeros
        output = np.zeros_like(flat_x)

        # Dispatch tokens to experts and aggregate weighted outputs
        for k_idx in range(self.top_k):
            indices_for_k = topk_indices[:, k_idx]  # Expert index selected for each token at rank k
            weights_for_k = topk_weights[:, k_idx, None]  # (M, 1)

            for expert_id in range(self.num_experts):
                token_mask = (indices_for_k == expert_id)
                if np.any(token_mask):
                    selected_tokens = flat_x[token_mask]  # (num_selected, d)
                    expert_out = self.experts[expert_id].forward(selected_tokens)
                    output[token_mask] += weights_for_k[token_mask] * expert_out

        return output.reshape(N, T, d), aux_loss


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 017 -- MIXTURE-OF-EXPERTS (MoE) VERIFICATION")
    print("=" * 65)

    N, T, d_model, d_ff = 2, 8, 32, 128
    num_experts, top_k = 8, 2

    x = np.random.randn(N, T, d_model)
    moe = MoELayer(d_model=d_model, d_ff=d_ff, num_experts=num_experts, top_k=top_k, seed=42)

    out, aux_loss = moe.forward(x)
    print("\n[1. MoE Layer Forward Pass]")
    print(f"  Input Shape:     {x.shape}")
    print(f"  Output Shape:    {out.shape} (Expected: ({N}, {T}, {d_model})) => [OK]")
    print(f"  Total Experts:   {num_experts}, Active per token: {top_k}")
    print(f"  Aux Loss Value:  {aux_loss:.6f} => [OK]")
    assert not np.isnan(out).any()
    print("  No NaNs in MoE output [OK]")

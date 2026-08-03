"""
AI ENGINEERING ACADEMY — MODULE 003 ENGINEERING CHALLENGE SOLUTION
Custom Softmax Cross-Entropy Computational Graph Node with Analytical Backward Pass
"""

import numpy as np

class SoftmaxCrossEntropyNode:
    """
    Numerically stable Softmax Cross-Entropy loss node with closed-form gradient.
    """
    def __init__(self):
        self.z = None
        self.y = None
        self.p = None
        self.loss = None
        self.grad_z = None

    def forward(self, logits, targets):
        """
        logits: np.array of shape (N, K)
        targets: one-hot np.array of shape (N, K)
        """
        self.z = np.atleast_2d(logits)
        self.y = np.atleast_2d(targets)

        # Numerical stability shift: z_shift = z - max(z)
        z_shift = self.z - np.max(self.z, axis=1, keepdims=True)
        exp_z = np.exp(z_shift)
        self.p = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        # Cross entropy loss: -sum(y * log(p + 1e-12))
        eps = 1e-12
        self.loss = -np.mean(np.sum(self.y * np.log(self.p + eps), axis=1))
        return self.loss

    def backward(self):
        """Analytical gradient: dL/dz = (p - y) / N"""
        N = self.z.shape[0]
        self.grad_z = (self.p - self.y) / N
        return self.grad_z


def verify_softmax_cross_entropy():
    print("=" * 65)
    print("MODULE 003 CHALLENGE SOLUTION: SOFTMAX CROSS-ENTROPY NODE")
    print("=" * 65)

    node = SoftmaxCrossEntropyNode()
    np.random.seed(42)
    logits = np.random.randn(4, 3)
    targets = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])

    loss = node.forward(logits, targets)
    analytical_g = node.backward()

    print(f"\nForward Loss: {loss:.6f}")
    print(f"Probabilities P:\n{np.round(node.p, 4)}")
    print(f"Analytical Gradients dL/dz:\n{np.round(analytical_g, 4)}")

    # Numerical gradcheck
    eps = 1e-5
    numerical_g = np.zeros_like(logits)

    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            orig_val = logits[i, j]

            logits[i, j] = orig_val + eps
            loss_plus = node.forward(logits, targets)

            logits[i, j] = orig_val - eps
            loss_minus = node.forward(logits, targets)

            logits[i, j] = orig_val
            numerical_g[i, j] = (loss_plus - loss_minus) / (2 * eps)

    # Re-run forward pass to restore state
    node.forward(logits, targets)

    rel_error = np.max(np.abs(analytical_g - numerical_g) / (np.maximum(np.abs(analytical_g), np.abs(numerical_g)) + 1e-8))
    print(f"\nGradcheck Relative Error: {rel_error:.2e}")
    print(f"Result: {'GRADCHECK PASSED [OK]' if rel_error < 1e-5 else 'FAILED'}")
    print("=" * 65)


if __name__ == "__main__":
    verify_softmax_cross_entropy()

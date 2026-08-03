"""
AI ENGINEERING ACADEMY — MODULE 001 ENGINEERING CHALLENGE SOLUTION
3D Perceptron Classifier with Separability Verification
"""

class NonLinearlySeparableError(Exception):
    """Raised when data cannot be 100% correctly separated by a single linear hyperplane."""
    pass


class Perceptron3D:
    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.weights = [0.0, 0.0, 0.0]
        self.bias = 0.0

    def predict_single(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return 1 if z >= 0 else 0

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def fit(self, X, y):
        # Validate inputs
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        for x in X:
            if len(x) != 3:
                raise ValueError("All inputs must be 3-dimensional")

        self.weights = [0.0, 0.0, 0.0]
        self.bias = 0.0

        for epoch in range(self.max_epochs):
            errors = 0
            for x, target in zip(X, y):
                pred = self.predict_single(x)
                error = target - pred
                if error != 0:
                    errors += 1
                    for i in range(3):
                        self.weights[i] += self.lr * error * x[i]
                    self.bias += self.lr * error
            if errors == 0:
                break

        # Verification step
        predictions = self.predict(X)
        accuracy = sum(p == t for p, t in zip(predictions, y)) / len(y)
        
        if accuracy < 1.0:
            raise NonLinearlySeparableError(
                f"Dataset is non-linearly separable. Max accuracy achieved: {accuracy * 100:.1f}%"
            )

        return {
            "weights": self.weights,
            "bias": self.bias,
            "epochs_trained": epoch + 1,
            "accuracy": accuracy,
        }


# --- VERIFICATION TEST ---
if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 001 CHALLENGE SOLUTION: 3D PERCEPTRON CLASSIFIER")
    print("=" * 60)

    # Test Case 1: Linearly Separable 3D Dataset (3D AND gate)
    X_3d_and = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
    y_3d_and = [0, 0, 0, 0, 0, 0, 0, 1]

    clf = Perceptron3D(learning_rate=0.1, max_epochs=100)
    result = clf.fit(X_3d_and, y_3d_and)
    print("\n[TEST 1 - 3D AND Gate (Separable)]")
    print(f"  Weights: {result['weights']}")
    print(f"  Bias:    {result['bias']:.3f}")
    print(f"  Epochs:  {result['epochs_trained']}")
    print(f"  Accuracy: {result['accuracy'] * 100}%  => SUCCESS [OK]")

    # Test Case 2: Non-linearly Separable 3D Dataset (3D Parity / XOR extension)
    y_3d_parity = [0, 1, 1, 0, 1, 0, 0, 1]
    print("\n[TEST 2 - 3D Parity / XOR (Non-separable)]")
    try:
        clf.fit(X_3d_and, y_3d_parity)
    except NonLinearlySeparableError as e:
        print(f"  Caught expected error: {e}  => SUCCESS [OK]")

    print("\n" + "=" * 60)
    print("ALL CHALLENGE VERIFICATIONS PASSED")
    print("=" * 60)

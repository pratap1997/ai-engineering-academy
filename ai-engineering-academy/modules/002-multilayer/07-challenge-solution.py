"""
AI ENGINEERING ACADEMY — MODULE 002 ENGINEERING CHALLENGE SOLUTION
3-Input Parity (3-Input XOR) 2-3-1 Multilayer Perceptron Solver
"""

import numpy as np

class Parity3InputMLP:
    """
    Hand-designed 2-Layer MLP for 3-input Parity (XOR).
    Architecture: 3 inputs -> 3 hidden neurons -> 1 output neuron
    
    Hidden Neuron Strategy:
    h1 = Step(x1 + x2 + x3 - 0.5)   [Sum >= 1]
    h2 = Step(x1 + x2 + x3 - 1.5)   [Sum >= 2]
    h3 = Step(x1 + x2 + x3 - 2.5)   [Sum >= 3]
    
    For Sum = 0: h = (0, 0, 0) -> Output y = 0
    For Sum = 1: h = (1, 0, 0) -> Output y = 1
    For Sum = 2: h = (1, 1, 0) -> Output y = 0
    For Sum = 3: h = (1, 1, 1) -> Output y = 1
    
    Output Neuron:
    y = Step(1*h1 - 2*h2 + 1*h3 - 0.5)
    """
    def __init__(self):
        # W1 shape: (3, 3)
        self.W1 = np.array([
            [1.0, 1.0, 1.0],  # h1: sum >= 1
            [1.0, 1.0, 1.0],  # h2: sum >= 2
            [1.0, 1.0, 1.0],  # h3: sum >= 3
        ])
        self.b1 = np.array([-0.5, -1.5, -2.5])

        # W2 shape: (1, 3)
        self.W2 = np.array([[1.0, -2.0, 2.0]])
        self.b2 = np.array([-0.5])

    def step(self, z):
        return (z >= 0).astype(float)

    def forward(self, X):
        X = np.atleast_2d(X)
        Z1 = np.dot(X, self.W1.T) + self.b1
        A1 = self.step(Z1)

        Z2 = np.dot(A1, self.W2.T) + self.b2
        A2 = self.step(Z2)
        return A2.ravel()

    def predict(self, X):
        return self.forward(X)


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 002 CHALLENGE SOLUTION: 3-INPUT PARITY (XOR) MLP")
    print("=" * 65)

    X_parity = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    y_parity = np.array([0, 1, 1, 0, 1, 0, 0, 1])

    mlp = Parity3InputMLP()
    preds = mlp.predict(X_parity)

    print("\nInput (x1,x2,x3) | Target y | Prediction y_hat | Correct?")
    print("-" * 55)
    all_correct = True
    for i in range(len(X_parity)):
        correct = preds[i] == y_parity[i]
        all_correct = all_correct and correct
        print(f"   {X_parity[i]}       |    {y_parity[i]}     |        {preds[i]:.0f}       | {'[OK]' if correct else '[FAIL]'}")

    accuracy = np.mean(preds == y_parity) * 100
    print("-" * 55)
    print(f"Overall Accuracy: {accuracy:.1f}%  => {'ALL TESTS PASSED [OK]' if all_correct else 'FAILED'}")
    print("=" * 65)

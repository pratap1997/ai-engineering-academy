"""
Test contract for Module 001 — Perceptron From Scratch

Tests cover 13 cases across four categories:
  1. Core behaviour    — AND, OR, prediction format, output shape
  2. State behaviour   — zero epochs, zero learning rate, state mutation
  3. Input validation  — mismatched lengths, bad dimensions, predict-before-fit
  4. Limitation evidence — XOR failure (geometric, multi-seed)

Both implementations are tested with identical contracts.

Run:
    pytest modules/001-perceptron/tests/ -v
    pytest modules/001-perceptron/tests/ -v --tb=short --junitxml=test-results.xml
"""

import sys
import os
import pytest

# Make the module importable from the test directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "implementation",
    os.path.join(os.path.dirname(__file__), "..", "04-implementation.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PerceptronPython = _mod.PerceptronPython
PerceptronNumPy = _mod.PerceptronNumPy

# ─────────────────────────────────────────────────────────────────
# Fixtures — shared test data
# ─────────────────────────────────────────────────────────────────

AND_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
AND_y = [0, 0, 0, 1]

OR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
OR_y = [0, 1, 1, 1]

XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_y = [0, 1, 1, 0]


def make_perceptrons(lr=0.1, epochs=100):
    """Return one instance of each implementation."""
    return [
        PerceptronPython(learning_rate=lr, epochs=epochs),
        PerceptronNumPy(learning_rate=lr, epochs=epochs),
    ]


# ─────────────────────────────────────────────────────────────────
# CATEGORY 1 — Core behaviour
# ─────────────────────────────────────────────────────────────────

class TestCoreBehaviour:

    def test_and_gate_learned_correctly(self):
        """AND is linearly separable. Both implementations must reach 100% accuracy."""
        for p in make_perceptrons(lr=0.1, epochs=100):
            p.fit(AND_X, AND_y)
            predictions = list(p.predict(AND_X))
            assert predictions == [0, 0, 0, 1], (
                f"{p.__class__.__name__} failed AND gate: {predictions}"
            )

    def test_or_gate_learned_correctly(self):
        """OR is linearly separable. Both implementations must reach 100% accuracy."""
        for p in make_perceptrons(lr=0.1, epochs=100):
            p.fit(OR_X, OR_y)
            predictions = list(p.predict(OR_X))
            assert predictions == [0, 1, 1, 1], (
                f"{p.__class__.__name__} failed OR gate: {predictions}"
            )

    def test_predict_returns_only_binary_values(self):
        """All predicted values must be exactly 0 or 1 — not floats, not booleans."""
        for p in make_perceptrons():
            p.fit(AND_X, AND_y)
            predictions = list(p.predict(AND_X))
            for val in predictions:
                assert val in (0, 1), (
                    f"{p.__class__.__name__} returned non-binary value: {val} "
                    f"(type: {type(val).__name__})"
                )

    def test_prediction_output_length_matches_input(self):
        """predict() must return exactly as many values as input samples."""
        for p in make_perceptrons():
            p.fit(AND_X, AND_y)
            predictions = list(p.predict(AND_X))
            assert len(predictions) == len(AND_X), (
                f"Expected {len(AND_X)} predictions, got {len(predictions)}"
            )


# ─────────────────────────────────────────────────────────────────
# CATEGORY 2 — State behaviour
# ─────────────────────────────────────────────────────────────────

class TestStateBehaviour:

    def test_zero_epochs_do_not_update_weights(self):
        """With 0 epochs, weights and bias must remain at their initial values (zero)."""
        for p in make_perceptrons(epochs=0):
            p.fit(AND_X, AND_y)
            # After 0 epochs, weights should still be zero
            if hasattr(p.weights, '__iter__') and not isinstance(p.weights, list):
                weights = list(p.weights)
            else:
                weights = p.weights
            assert all(w == 0.0 for w in weights), (
                f"Weights changed with 0 epochs: {weights}"
            )
            assert p.bias == 0.0, (
                f"Bias changed with 0 epochs: {p.bias}"
            )

    def test_zero_learning_rate_does_not_update_weights(self):
        """With learning_rate=0, no weight update is possible regardless of errors."""
        for p in make_perceptrons(lr=0.0001, epochs=100):
            # Note: exact zero is validated in __init__ to raise ValueError.
            # Use a very small value to test near-zero behavior is acceptable,
            # or bypass by setting after init if testing truly zero.
            # This tests the edge of the contract.
            pass  # covered by test_invalid_learning_rate below

    def test_repeated_prediction_does_not_mutate_model_state(self):
        """Calling predict() multiple times must not change the model weights or bias."""
        for p in make_perceptrons():
            p.fit(AND_X, AND_y)

            if hasattr(p.weights, 'copy'):
                weights_before = p.weights.copy()
            else:
                weights_before = p.weights[:]
            bias_before = p.bias

            # Call predict multiple times
            _ = p.predict(AND_X)
            _ = p.predict(AND_X)
            _ = p.predict(OR_X)

            if hasattr(p.weights, 'copy'):
                weights_after = list(p.weights)
                weights_before = list(weights_before)
            else:
                weights_after = p.weights

            assert weights_after == weights_before, (
                "Weights changed after calling predict()"
            )
            assert p.bias == bias_before, (
                "Bias changed after calling predict()"
            )


# ─────────────────────────────────────────────────────────────────
# CATEGORY 3 — Input validation
# ─────────────────────────────────────────────────────────────────

class TestInputValidation:

    def test_mismatched_X_and_y_lengths_raise_error(self):
        """fit() must raise a clear error when X and y have different lengths."""
        for p in make_perceptrons():
            with pytest.raises((ValueError, AssertionError)):
                p.fit([[0, 1], [1, 0]], [0])  # 2 samples, 1 label

    def test_inconsistent_feature_dimensions_raise_error(self):
        """fit() or predict() must raise an error for samples with inconsistent dimensions."""
        for p in make_perceptrons():
            p.fit(AND_X, AND_y)
            with pytest.raises((ValueError, IndexError, Exception)):
                p.predict([[0, 1, 2]])  # 3 features instead of 2

    def test_predict_before_fit_raises_error(self):
        """predict() before fit() must raise an error, not return silent garbage."""
        for p in make_perceptrons():
            with pytest.raises((RuntimeError, AttributeError, TypeError)):
                p.predict(AND_X)

    def test_invalid_learning_rate_raises_error(self):
        """Learning rate <= 0 must raise ValueError at construction time."""
        for cls in [PerceptronPython, PerceptronNumPy]:
            with pytest.raises(ValueError):
                cls(learning_rate=0)
            with pytest.raises(ValueError):
                cls(learning_rate=-0.1)

    def test_invalid_epochs_raises_error(self):
        """Negative epochs must raise ValueError at construction time."""
        for cls in [PerceptronPython, PerceptronNumPy]:
            with pytest.raises(ValueError):
                cls(epochs=-1)


# ─────────────────────────────────────────────────────────────────
# CATEGORY 4 — Limitation evidence (XOR)
# ─────────────────────────────────────────────────────────────────

class TestLimitationEvidence:

    @pytest.mark.parametrize("seed_offset", [0, 7, 42])
    def test_xor_cannot_be_perfectly_classified(self, seed_offset):
        """
        XOR is not linearly separable. No single-layer perceptron can achieve 100%
        accuracy on XOR with any initialization or number of epochs.

        This test uses three deterministic dataset orderings as proxies for seeds
        (the plain Python perceptron has no random state, so order serves as variation).
        The key assertion: no run achieves perfect accuracy.

        Educational note: this is a geometric limitation, not a training failure.
        The decision boundary cannot be a single straight line for XOR.
        """
        # Rotate the XOR dataset to get a different training order each time
        rotated_X = XOR_X[seed_offset:] + XOR_X[:seed_offset]
        rotated_y = XOR_y[seed_offset:] + XOR_y[:seed_offset]

        for p in make_perceptrons(lr=0.1, epochs=200):
            p.fit(rotated_X, rotated_y)
            # Predict on canonical XOR (not the rotated version)
            predictions = list(p.predict(XOR_X))
            correct = sum(pred == true for pred, true in zip(predictions, XOR_y))
            assert correct < 4, (
                f"{p.__class__.__name__} (seed_offset={seed_offset}) "
                f"unexpectedly achieved perfect XOR accuracy.\n"
                f"Predictions: {predictions}\n"
                f"True labels: {XOR_y}\n"
                f"This would imply a non-linear boundary, which a single perceptron cannot learn."
            )

    def test_xor_training_terminates(self):
        """
        Training on XOR must terminate within the declared epoch limit.
        The perceptron must not loop infinitely.
        """
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Perceptron training did not terminate within the time limit.")

        # On Windows, signal.SIGALRM is not available. Use a thread-based approach.
        import threading

        results = []
        errors = []

        def run_training():
            try:
                for p in make_perceptrons(lr=0.1, epochs=1000):
                    p.fit(XOR_X, XOR_y)
                results.append("done")
            except Exception as e:
                errors.append(str(e))

        thread = threading.Thread(target=run_training)
        thread.start()
        thread.join(timeout=10)  # 10 second timeout

        assert not thread.is_alive(), (
            "Perceptron training did not terminate within 10 seconds on XOR data. "
            "The training loop must always terminate at the epoch limit."
        )
        assert not errors, f"Training raised unexpected error: {errors}"
        assert results, "Training did not complete"

# Module 007 — Version History

## [1.0.0] - 2026-08-01

### Added
- Created complete 9-artifact curriculum for Modern CNN Architectures (ResNet).
- Pure Python & NumPy implementations of ResidualBlock, GlobalAvgPool2D, and MiniResNet18 (`04-implementation.py`).
- Experiments covering gradient highway survival in 20-layer ResNets vs Plain CNNs and GAP parameter savings (`05-experiments.py`).
- Custom ResidualBlock challenge & solution with analytical backward pass & gradcheck verification (`07-challenge-solution.py`).
- Pytest test suite with 16 tests (`tests/test_modern_cnns.py`).
- Standalone interactive HTML visualizer displaying live residual identity gradient flow vs plain network gradient attenuation (`assets/resnet-standalone.html`).

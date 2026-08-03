# Module 022: Engineering Challenge — Multi-Layer Student Distillation Pipeline

## 1. Challenge Task

Construct a self-contained `MultiLayerStudentDistiller` in pure Python & NumPy that:
1. Implements a 2-layer Deep Student model learning from a 4-layer Deep Teacher model.
2. Extracts both **Soft Logit Distillation Loss** ($T=4.0$) and **Intermediate Hidden Feature Alignment Loss** ($\text{MSE}(H_{\text{student}}, H_{\text{teacher}} \mathbf{W}_{\text{proj}})$).
3. Trains over 40 iterations, logging total loss, KL loss, and student accuracy.
4. Verifies that feature-aligned student matches $>95\%$ of Teacher prediction accuracy.

---

## 2. Validation Criteria

1. Student model accuracy reaches $>95\%$ relative to Teacher accuracy.
2. KL loss and feature alignment MSE decrease monotonically.
3. Zero NaNs or gradient explosions.

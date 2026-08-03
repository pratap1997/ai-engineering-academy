# Module 016: Mental Model — The Grid Ruler & Zero-Point Alignment

## 1. The Grid Ruler Analogy

Think of quantization as snapping continuous measurement values to tick marks on a ruler:

```
Continuous FP32 values:   -1.42    -0.12    0.87    1.95
Ruler tick marks (INT8):   -128 ... -10  ... 45  ... 127
```

- **Scale ($s$)**: The distance between two adjacent tick marks on the ruler.
- **Zero-Point ($z$)**: The integer tick mark that corresponds to the real-world value $0.0$.

If tick marks are too far apart (low bit width like INT2), rounding error is large.
If tick marks are placed intelligently per channel or per group of 128 weights, rounding error drops to near zero!

---

## 2. Symmetric vs Asymmetric Mapping

```
Symmetric (z = 0):
Real Range: [-alpha, +alpha]  ──────→  INT8 Range: [-127, +127]
  - Symmetric around 0.0
  - One calculation: q = round(x / s)

Asymmetric (z != 0):
Real Range: [min, max]        ──────→  INT8 Range: [0, 255]
  - Shifts zero so full 8-bit range [0, 255] is utilized
  - Two calculations: q = round(x / s) + z
```

---

## 3. Per-Tensor vs Per-Channel vs Group-wise Quantization

- **Per-Tensor**: Single scale $s$ for the entire weight matrix. Outliers in one row cause loss of precision everywhere else.
- **Per-Channel (Per-Row)**: Separate scale $s_i$ for each row of the weight matrix. Drastically reduces outlier impact.
- **Group-wise (Block-wise)**: Partition each row into sub-blocks of 64 or 128 weights, each with its own scale $s_{i, g}$. Standard for INT4 (GPTQ / AWQ).

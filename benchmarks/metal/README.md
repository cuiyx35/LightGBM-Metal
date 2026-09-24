# Apple M5 synthetic benchmark

These reports were produced by the experimental Metal source branch on an
Apple M5 Mac with 32 GiB unified memory, connected to AC power. The Metal
implementation was at commit `885c799`. Both runs used the Metal-enabled
LightGBM 4.7.0 source checkout and the same preconstructed Dataset.
After the benchmark, a fresh CMake configure and build directory succeeded
with CMake 4.4.3 and Apple Clang 21; the four validation cases passed again
against the rebuilt library.

| Report | Workload | Result |
| --- | --- | --- |
| [Scale benchmark](m5_synthetic_4m_500trees.json) | 3.3 million fit rows, 0.8 million held rows, 512 features, 500 trees, 6 CPU threads | CPU/Metal median fit ratio **1.55×** |
| [Model-level validation](m5_validation.json) | Four small synthetic cases | All passed |

The large benchmark generated every feature and label from its fixed seed.
It used 150 dense numeric and 362 rare binary features. After one-tree CPU
and Metal warmups, the run order was Metal → CPU → CPU → Metal. Fit times in
seconds were Metal **108.00 / 125.67** and CPU **178.93 / 183.50**. Median
fit plus prediction was Metal **121.35** versus CPU **185.85** seconds,
or **1.53×**. Generation and Dataset construction took **21.72** seconds
once and were excluded from those per-backend times. Adding that same shared
time to each backend gives an estimated full ratio of **1.45×**. The
process peak RSS was **9.72 GiB**.

Repeated runs of each backend produced identical model and prediction hashes.
The maximum Metal/CPU difference on held predictions was **1.86e-6**;
AP differed by **2.07e-9** on these generated labels. This says nothing
about the quality or prediction differences of application models. The
synthetic data do not reproduce real feature distributions, missingness,
categorical values, or label relationships. This is one chip, one seed, and
two runs per backend; thermal conditions and background load can change
timings. The validation case times are not a speed benchmark.

To reproduce after building and selecting this fork's Python package and
Metal-enabled native library:

```bash
python examples/python-guide/metal_validate.py \
  --output metal_validation.json
python examples/python-guide/metal_synthetic_benchmark.py \
  --output metal_synthetic_benchmark.json
```

The scripts write only aggregate JSON and hashes. No external data files
or row-level predictions are included here.

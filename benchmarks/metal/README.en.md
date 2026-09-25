# Apple M5 synthetic benchmark

[简体中文](README.md) · **English** · [Project home](../../README.en.md)

These reports were produced by the experimental Metal fork on an
Apple M5 Mac with 32 GiB unified memory, connected to AC power. The Metal
implementation in the scale benchmark and first profile used the source state
represented by commit `7807af1` after the standalone history rewrite. The
GPU execution follow-up added diagnostic timing to the same kernel. Runs used the Metal-enabled
LightGBM 4.7.0 source checkout and the same preconstructed Dataset.
After the benchmark, a fresh CMake configure and build directory succeeded
with CMake 4.4.3 and Apple Clang 21; the five validation cases passed again
against the rebuilt library.

| Report | Workload | Result |
| --- | --- | --- |
| [Scale benchmark](m5_synthetic_4m_500trees.json) | 3.3 million fit rows, 0.8 million held rows, 512 features, 500 trees, 6 CPU threads | CPU/Metal median fit ratio **1.55×** |
| [Model-level validation](m5_validation.json) | Five small synthetic cases | All passed |
| [Bottleneck profile](m5_profile_3m_100trees.json) | 3.3 million fit rows, 100 trees, 6 threads, profiling enabled | 16.14 s cumulative GPU command time; 12.92 s CPU wait for GPU |
| [GPU execution follow-up](m5_gpu_execution_3m_100trees.json) | A new diagnostic run with the same configuration | 14.58 s cumulative GPU execution reported by Metal; 15.87 s cumulative command in-flight time |
| [Large-leaf gradient staging comparison](m5_stage_selected_3m_100trees.json) | 3.3 million fit rows, 0.2 million generated held rows, 512 features, 100 trees, 6 threads; interleaved old/new Metal paths | **1.27×** old/new median fit-time ratio; identical model and prediction hashes |
| [Current CPU/Metal comparison](m5_current_3m_100trees_cpu_metal.json) | 3.3 million fit rows, 0.2 million generated held rows, 512 features, 100 trees, 6 threads; interleaved Metal/CPU runs | **1.67×** CPU/Metal median fit-time ratio |

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

The 500-tree CPU/Metal ratio above came from pre-optimization commit `7807af1`;
it is not a 500-tree speed measurement of the current branch. The current
branch stages and quantizes gradients and Hessians on the GPU for leaves
larger than one shard by default. The new Metal-to-Metal comparison used
generated data with a fixed seed and ran old → new → new → old. Fit times
were **21.945 / 17.528 / 17.697 / 22.851** seconds; old/new medians were
**22.398 / 17.612** seconds, a ratio of about **1.27×**. It does not include
data generation, Dataset construction, or prediction, and it is not a new
CPU/Metal speed ratio. Model text and held prediction hashes matched exactly
across modes on this generated task. Thermals and background load can still
change the timings.

The current version's 100-tree CPU/Metal comparison used the same generator
and Dataset, in Metal → CPU → CPU → Metal order. Metal fit times were
**17.253 / 18.225** seconds and CPU fit times were **28.941 / 30.296**
seconds. Median fit times were **17.739 / 29.618** seconds, a CPU/Metal
ratio of about **1.67×**. Median fit plus prediction ratio was about
**1.66×**. Maximum Metal/CPU held prediction difference was **3.17e-6**;
none exceeded 0.001. This test has fewer trees and a smaller held set than
the historical 500-tree test, so the two ratios do not measure a
cross-version speed change. Neither test establishes application model
quality.

To reproduce after building and selecting this fork's Python package and
Metal-enabled native library:

```bash
python examples/python-guide/metal_validate.py \
  --output metal_validation.json
python examples/python-guide/metal_synthetic_benchmark.py \
  --full-scale --output metal_synthetic_benchmark.json
python examples/python-guide/metal_stage_benchmark.py \
  --full-scale --output metal_stage_benchmark.json
python examples/python-guide/metal_synthetic_benchmark.py \
  --train-rows 3300000 --held-rows 200000 --features 512 \
  --dense-features 150 --rounds 100 --threads 6 --repeats 2 \
  --output metal_current_cpu_metal.json
```

The scripts write only aggregate JSON and hashes. No external data files
or row-level predictions are included here.

### Bottleneck profile

The profiles use the same generator but train only 100 trees, with extra
timing enabled. In the first run, tree learner training accumulated 19.58 s;
GPU commands were in flight for 16.14 s. CPU histograms took 3.22 s,
split search 0.49 s, and the data mirror 0.62 s. In a later run with GPU
timestamps, all 6,200 commands returned valid timings: Metal reported
14.58 s cumulative GPU execution within 15.87 s cumulative in-flight wall
time. These separate runs cannot be subtracted item by item. Some CPU and GPU
work overlaps, and GPU wait is contained in command in-flight time. **Do not
add these times.** This workload points to the GPU histogram kernel as the
next performance target. Neither profile is a speed result or represents
other data. Use the unprofiled interleaved benchmark above for speed
comparisons.

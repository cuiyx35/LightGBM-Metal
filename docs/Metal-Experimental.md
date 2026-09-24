# Experimental Metal training on Apple Silicon

This branch is an independent research fork of LightGBM 4.7.0. It adds
`device_type="metal"` for macOS on Apple Silicon. It is not an official
LightGBM release, and it has not been validated across M-series models.

The Metal path calculates histograms for eligible feature groups. The CPU
calculates the other groups at the same time, then performs split search and
the rest of tree training. Eligible groups currently contain one feature,
are not multi-value groups, and have no more than 256 bins. Prediction uses
the standard CPU model path. Metal histogram sums use fixed-point
quantization, so the resulting trees and predictions can differ from CPU
training. Test application quality separately before deploying a model.

## Build and use

Building needs macOS, Apple Silicon, a Metal-capable GPU, CMake, a C++
toolchain, and a working OpenMP runtime for parallel CPU training. The
Python benchmark and validation scripts also need NumPy, pandas, and
scikit-learn. Configure
OpenMP as required by your local toolchain. A typical source build starts
with:

```bash
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DUSE_METAL=ON
cmake --build build-metal -j2
```

Build or install the Python package from this checkout, and verify that it
loads the Metal-enabled `lib_lightgbm.dylib`. For example:

```python
import lightgbm as lgb

params = {"objective": "binary", "device_type": "metal", "num_threads": 6}
model = lgb.train(params, training_dataset, num_boost_round=500)
model.save_model("model.txt")
```

To return to the CPU path, set `device_type="cpu"`. Model prediction and
serialization use the usual LightGBM interfaces.

## Scriptable validation

[`examples/python-guide/metal_validate.py`](../examples/python-guide/metal_validate.py)
runs four synthetic model-level cases: dense numeric, mixed missing/category/
sparse features with bagging, multiple histogram shards, and a constant
Hessian CPU fallback. It checks CPU/Metal predictions and model reload,
then writes a JSON report. A failed check exits nonzero and writes
`"status": "FAIL"`. This is a smoke test, not a rule requiring application
models to reproduce CPU predictions exactly.

```bash
python examples/python-guide/metal_validate.py --output metal_validation.json
```

The two scripts provide command-line arguments, exit codes, and JSON output
for CI or AI-assisted integration. LightGBM already has its own training CLI
and Python callbacks; a separate monitoring dashboard is not required to
run or inspect this experiment.

## Public synthetic benchmark

[`examples/python-guide/metal_synthetic_benchmark.py`](../examples/python-guide/metal_synthetic_benchmark.py)
generates all its rows and labels locally from a fixed seed. Its default
workload is 3.3 million fit rows, 0.8 million held rows, 512 features (150
dense numeric and 362 rare binary), and 500 boosting rounds. It alternates two
Metal and two CPU runs after one-tree warmups, using the same constructed
Dataset. It reports timing, AP/AUC on synthetic labels, prediction differences,
repeated-run hashes, and process peak RSS. No external data file is read and
the report contains no row-level examples or predictions.

```bash
python examples/python-guide/metal_synthetic_benchmark.py \
  --threads 6 --output synthetic_metal_result.json
```

For a smaller smoke test, use `--train-rows 30000 --held-rows 5000
--features 40 --dense-features 12 --rounds 10`. The default full benchmark
needs substantial unified memory; the source matrix alone occupies about
2 GiB. The generated feature distribution, feature bundling, missingness,
and label relationships will differ from real application data, even when
row count and feature count match. Its quality metrics measure only the
generated task. Speed also depends on chip model, thermal conditions,
background load, and OpenMP configuration.

One Apple M5 run of the default workload and the four validation cases are
recorded in [the public benchmark reports](../benchmarks/metal/README.md).

This fork retains the upstream [MIT license](../LICENSE) and copyright
notices; third-party submodules retain their own license files. Benchmark
results should state the exact commit, hardware,
configuration, run order, and whether data construction is included.

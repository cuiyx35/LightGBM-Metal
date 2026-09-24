# Experimental Metal training on Apple Silicon

This repository is an independent research fork of LightGBM 4.7.0. It adds
`device_type="metal"` for macOS on Apple Silicon. It is not an official
LightGBM release, and it has not been validated across M-series models.
The standalone repository history starts with a source snapshot of upstream
commit [`8f7036f`](https://github.com/lightgbm-org/LightGBM/commit/8f7036f03627054d5a54a6f965b13f4b9ff2cb63);
earlier LightGBM history remains in the official repository.
For architecture and contributor workflow, read the
[Metal development guide](Metal-Development.md). Coding agents can also use
the repository [AGENTS.md](../AGENTS.md).

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
ln -sfn ../lib_lightgbm.dylib python-package/lib_lightgbm.dylib
export PYTHONPATH="$PWD/python-package"
```

If OpenMP is installed outside standard library paths, also add its `lib/`
directory to `DYLD_LIBRARY_PATH` for Python runs. Verify that Python loads
the package and Metal-enabled `lib_lightgbm.dylib` from this checkout. For
example:

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
runs five synthetic model-level cases: dense numeric, mixed missing/category/
sparse features with bagging, multiple histogram shards, a constant
Hessian CPU fallback, and CPU/GPU overlap versus serial execution. It checks
CPU/Metal predictions, overlap, and model reload,
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
generates all its rows and labels locally from a fixed seed. By default it
runs a quick 30,000-fit-row smoke test. Explicit `--full-scale` selects 3.3
million fit rows, 0.8 million held rows, 512 features (150 dense numeric and
362 rare binary), and 500 boosting rounds. The full-scale preset alternates two
Metal and two CPU runs after one-tree warmups, using the same constructed
Dataset. It reports timing, AP/AUC on synthetic labels, prediction differences,
repeated-run hashes, and process peak RSS. No external data file is read and
the report contains no row-level examples or predictions.

```bash
python examples/python-guide/metal_synthetic_benchmark.py \
  --full-scale --output synthetic_metal_result.json
```

For a small smoke test, omit `--full-scale`. The full-scale benchmark
needs substantial unified memory; the source matrix alone occupies about
2 GiB. The generated feature distribution, feature bundling, missingness,
and label relationships will differ from real application data, even when
row count and feature count match. Its quality metrics measure only the
generated task. Speed also depends on chip model, thermal conditions,
background load, and OpenMP configuration.

One Apple M5 run of the full-scale preset and the five validation cases are
recorded in [the public benchmark reports](../benchmarks/metal/README.md).

This fork retains the upstream [MIT license](../LICENSE) and copyright
notices; third-party submodules retain their own license files. Benchmark
results should state the exact commit, hardware, configuration, run order,
and whether data construction is included.

# LightGBM Metal: an Apple Silicon experiment

[简体中文](README.md) · **English**

> [!IMPORTANT]
> This is an independent experimental fork based on LightGBM 4.7.0, not an official LightGBM release. The Metal backend has been performance-tested on one Apple M5 only. Validate model quality on your own data before practical use.

This project adds <code>device_type="metal"</code> on Apple Silicon. During training, the GPU calculates histograms for eligible feature groups while the CPU calculates histograms for the remaining groups. The CPU also performs split search and grows trees. Prediction and model saving use the standard LightGBM paths. Training **combines CPU and GPU work**; it is not a GPU-only backend.

## Current capabilities and limits

- The Metal path handles feature groups with one feature, no multi-value representation, and at most 256 bins. Other groups run on the CPU.
- GPU histogram sums use fixed-point quantization. Close split decisions can produce different trees and predictions from CPU training. Small prediction differences or similar metrics on synthetic data do not establish application model quality.
- Building requires Apple Silicon macOS, a Metal-capable GPU, CMake, a C++ toolchain, and a working OpenMP runtime. Python validation scripts also require NumPy, pandas, and scikit-learn.
- Performance has not been checked on other M-series chips. Speed depends on data distribution, chip, CPU thread count, thermals, and background load.

See the [experimental user notes](docs/Metal-Experimental.en.md) for architecture, limits, and options. Contributors should read the [development guide](docs/Metal-Development.en.md).

## Build and quick validation

Run from the repository root. Install CMake and an OpenMP runtime, plus NumPy, SciPy, pandas, scikit-learn, and narwhals for validation. With Homebrew, the script finds <code>libomp</code> automatically:

~~~bash
brew install cmake libomp
bash tools/build-metal-macos.sh --validate
~~~

The script builds in <code>build-metal/</code> and links the native library into this checkout's Python package. <code>--validate</code> checks five model-level cases using generated data, writes <code>build-metal/metal_validation.json</code>, and exits nonzero on failure. Set <code>OPENMP_PREFIX</code> for a non-Homebrew OpenMP installation or <code>PYTHON_BIN</code> for another Python. Run <code>bash tools/build-metal-macos.sh --help</code> for all options. The [Metal macOS GitHub Actions check](.github/workflows/metal_macos.yml) uses the same script on an Apple Silicon hosted runner; it does not run a performance benchmark.

For a separate small performance smoke test after building:

~~~bash
PYTHONPATH="$PWD/python-package" python3 examples/python-guide/metal_synthetic_benchmark.py --output metal_quick_benchmark.json
~~~

Set <code>device_type="metal"</code> to train with Metal, or <code>device_type="cpu"</code> for the CPU path.
For repeated fits with the same training rows, feature order, categorical definitions, and binning parameters, construct one <code>lightgbm.Dataset</code> and reuse it with <code>lightgbm.train()</code>. Rebuild it when changing <code>max_bin</code> or the cross-validation split. See the [usage example](docs/Metal-Experimental.en.md#reuse-a-dataset-across-fits).

## Public benchmark

On one Apple M5 with 32 GiB unified memory, a historical generated-data test with 3.3 million fit rows, 0.8 million held rows, 512 features, 500 trees, and 6 CPU threads produced a **1.55× CPU/Metal median training-time ratio**. The median ratio for training plus prediction was 1.53×. Including the one-time data generation and Dataset construction cost shared by both backends gives an estimated overall ratio of 1.45×. **This result came from commit <code>7807af1</code>, before large-leaf gradient staging.** The current version stages and quantizes selected gradients and Hessians on the GPU for large leaves by default. See the [public benchmark report](benchmarks/metal/README.en.md) for the Metal path comparison, raw JSON, and limitations. Results from one machine do not describe other chips or real application data.

A separate generated-data comparison with 3.3 million fit rows and 100 trees found a **1.27× old/new Metal median fit-time ratio**, with identical model and prediction hashes. This compares two Metal algorithms, not the current version against CPU.
In a paired CPU/Metal test at the same 100-tree scale, the current version had a **1.67× CPU/Metal median fit-time ratio**. This has a different tree count from the historical 500-tree test; see the report for details.

The current version also completed an interleaved 500-tree CPU/Metal comparison with 3.3 million fit rows and 0.2 million generated held rows. The median fit-time ratio in these runs was **1.89×**. Both backends' second runs were substantially slower, so this is not a stable speed guarantee, and it cannot be subtracted from the historical 500-tree result. The [benchmark report](benchmarks/metal/README.en.md) includes raw JSON, prediction differences, and limitations.

The full-scale synthetic test needs substantial unified memory and must be selected explicitly:

~~~bash
python examples/python-guide/metal_synthetic_benchmark.py \
  --full-scale --output synthetic_metal_result.json
~~~

For parameter comparisons, set <code>--max-bin</code>, <code>--num-leaves</code>, or <code>--feature-fraction</code>. To enable row sampling, set both <code>--bagging-fraction</code> and <code>--bagging-freq</code>. The script records the configuration, Dataset construction time, CPU/Metal timing, and generated-data metrics. Model quality under a new configuration requires separate validation.

## Optimization status

As of 2026-09-25, <code>main</code> includes concurrent CPU/GPU histogram construction, a default 32,768-row GPU shard, 64 threadgroups, dispatch of active feature groups only, and selected-gradient staging for large leaves. The results above and the [benchmark report](benchmarks/metal/README.md) cover the integrated backend. The [experimental guide](docs/Metal-Experimental.en.md) also explains how to reuse a Dataset across repeated fits.

GPU-resident tree construction and its categorical split handling remain standalone prototypes. This round produced no further default training-path change ready for integration.

## Star History

<a href="https://www.star-history.com/?repos=cuiyx35%2FLightGBM-Metal&type=date&legend=top-left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=cuiyx35/LightGBM-Metal&type=date&theme=dark&legend=top-left" />
    <img alt="LightGBM-Metal Star History" src="https://api.star-history.com/chart?repos=cuiyx35/LightGBM-Metal&type=date&legend=top-left" />
  </picture>
</a>

## Documentation and extension

| Topic | 中文 | English |
| --- | --- | --- |
| Project entry | [README.md](README.md) | This page |
| Metal usage and validation | [实验使用说明](docs/Metal-Experimental.md) | [Experimental Metal notes](docs/Metal-Experimental.en.md) |
| Architecture and extension workflow | [开发指南](docs/Metal-Development.md) | [Development guide](docs/Metal-Development.en.md) |
| Benchmarks and limitations | [公开基准报告](benchmarks/metal/README.md) | [Benchmark report](benchmarks/metal/README.en.md) |
| Coding-agent guidance | [AGENTS.md](AGENTS.md) | [AGENTS.en.md](AGENTS.en.md) |

## Provenance and license

This repository starts from a source snapshot of [official LightGBM 4.7.0 commit <code>8f7036f</code>](https://github.com/lightgbm-org/LightGBM/commit/8f7036f03627054d5a54a6f965b13f4b9ff2cb63). Earlier history remains in the [official LightGBM repository](https://github.com/lightgbm-org/LightGBM). The fork retains the upstream [MIT license](LICENSE) and third-party license notices. Use the [official documentation](https://lightgbm.readthedocs.io/) for generic LightGBM functionality and installation guidance.

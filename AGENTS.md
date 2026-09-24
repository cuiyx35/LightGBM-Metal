# Guidance for coding agents

This repository is an independent experimental LightGBM fork.
The Metal backend is experimental and runs only on Apple Silicon macOS.
Read [the Metal development guide](docs/Metal-Development.md) before changing
the backend. Keep upstream LightGBM behavior unchanged when `USE_METAL=OFF`.

## Work in this branch

- The entry points are `CMakeLists.txt`, `include/LightGBM/config.h`,
  `src/io/config.cpp`, `src/treelearner/tree_learner.cpp`, and
  `src/treelearner/metal_tree_learner.{h,mm}`.
- The Metal learner currently assigns whole feature groups to GPU or CPU.
  CPU and GPU build histograms concurrently; split search stays on CPU.
  Preserve that ownership rule when adding supported group types.
- Keep source, examples, and benchmark reports free of private datasets,
  private feature names, row-level predictions, credentials, and local paths.
  Retain upstream and third-party license notices.

## Check changes

1. Build with `-DUSE_METAL=ON` in a separate CMake build directory. On macOS,
   configure a working OpenMP runtime for parallel CPU training.
2. Run `examples/python-guide/metal_validate.py --output /tmp/metal_validation.json`.
   Its five synthetic cases check mixed feature handling, fallback, model
   reload, and CPU/GPU overlap. Confirm `status` is `PASS`.
3. Run `examples/python-guide/metal_synthetic_benchmark.py --output
   /tmp/metal_benchmark.json` for a quick CPU/Metal smoke test. Use
   `--full-scale` only for a planned, idle-machine performance comparison.
4. For any numerical or performance change, compare identical inputs and
   parameters, record CPU/Metal run order and warmups, inspect model and
   prediction hashes, and document both speed and quality differences.

The Python scripts write aggregate JSON and exit nonzero on failure. They
require this fork's Python package and Metal-enabled native library; verify
the imported paths before interpreting a benchmark. Profiling environment
variables add overhead and should be off for headline timing.

Do not push to the `upstream` remote, which points at official LightGBM.
For a public project, first add an `origin` remote under the owner's GitHub
account and review the target and commit identity. Public source benchmarks
must use generated data only.

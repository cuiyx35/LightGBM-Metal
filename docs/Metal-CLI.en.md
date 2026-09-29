# Metal experiment CLI and local dashboard

**English** · [简体中文](Metal-CLI.md)

Run these commands from the repository root with Python 3.10 or newer. The CLI wraps the existing build and synthetic-data checks. It does not load external datasets.

~~~bash
python3 tools/metal_experiment.py doctor --json
python3 tools/metal_experiment.py build --validate
python3 tools/metal_experiment.py benchmark
python3 tools/metal_experiment.py report
python3 tools/metal_experiment.py serve
~~~

`serve` binds only to `127.0.0.1:8765` and refreshes every three seconds. Run build, validation, or benchmark in another terminal to see command status and completed results. It does not show per-tree GPU utilization or internal training curves. `report` writes a standalone HTML page to `build-metal/metal_report.html` by default.

For an existing public report:

~~~bash
python3 tools/metal_experiment.py report \
  --benchmark benchmarks/metal/m5_current_3m_100trees_cpu_metal.json \
  --validation benchmarks/metal/m5_validation.json \
  --output build-metal/public_report.html
~~~

`doctor --json` prints one JSON object. `build`, `validate`, `benchmark`, and `wheel` accept `--json-out PATH` for a stable status record: `schema_version`, `command`, `status` (`RUNNING`, `PASS`, `FAIL`), `started_at`, `finished_at`, `message`, and `outputs`. During a run, the same state is updated in `build-metal/experiment_status.json`. Detailed timing and quality remain in the benchmark JSON. Failures return a nonzero exit code.

The default benchmark is small. `benchmark --full-scale` explicitly selects about 4.1 million generated rows and 512 features. Check machine load before using it. Workload controls include `--train-rows`, `--held-rows`, `--features`, `--dense-features`, `--rounds`, `--threads`, and `--repeats`.

## Experimental wheel

Building the fork from source still requires CMake, a C++ toolchain, and OpenMP. A wheel moves compilation to a build machine:

~~~bash
brew install cmake libomp
python3 -m pip install build
python3 tools/metal_experiment.py wheel
~~~

The build machine must run Apple Silicon macOS. On a compatible Apple Silicon Mac, installing the resulting wheel does not compile LightGBM, but `libomp` remains a runtime dependency:

~~~bash
brew install libomp
python3 -m pip install dist/lightgbm-*.whl
~~~

Use a separate virtual environment. This experimental wheel keeps the upstream `lightgbm` distribution name and 4.7.0 version, so it replaces the official LightGBM in that environment. Do not publish it to PyPI or present it as an official release. The Metal macOS GitHub Actions job builds it, installs it in a clean environment, runs a small Metal training check, and then uploads it as a workflow artifact. Check the run and commit before downloading. Do not commit private reports or datasets.

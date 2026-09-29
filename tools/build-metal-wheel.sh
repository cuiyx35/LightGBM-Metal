#!/bin/bash
# Build a machine-specific experimental Metal wheel. No upload or installation.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
  echo "Metal wheel builds require Apple Silicon macOS." >&2
  exit 2
}
command -v cmake >/dev/null 2>&1 || { echo "Install CMake before building a wheel." >&2; exit 2; }
command -v brew >/dev/null 2>&1 || { echo "This wheel builder currently requires Homebrew libomp." >&2; exit 2; }
openmp_prefix="$(brew --prefix libomp)" || { echo "Install libomp: brew install libomp" >&2; exit 2; }
[[ -f "$openmp_prefix/include/omp.h" && -f "$openmp_prefix/lib/libomp.dylib" ]] || {
  echo "libomp headers or library missing in $openmp_prefix" >&2
  exit 2
}
"$python_bin" -m build --version >/dev/null 2>&1 || {
  echo "Install the Python build frontend: $python_bin -m pip install build" >&2
  exit 2
}

cd "$repo_root"
PYTHON_BIN="$python_bin" sh ./build-python.sh bdist_wheel --metal

wheel="$(find "$repo_root/dist" -maxdepth 1 -name 'lightgbm-*.whl' -print -quit)"
[[ -n "$wheel" ]] || { echo "No wheel was produced." >&2; exit 1; }
"$python_bin" - "$wheel" <<'PY'
from pathlib import Path
import sys
import zipfile

wheel = Path(sys.argv[1])
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    if not any(name.endswith("/lib_lightgbm.dylib") for name in names):
        raise SystemExit("Wheel does not contain lib_lightgbm.dylib")
print(f"WHEEL_READY {wheel}")
PY

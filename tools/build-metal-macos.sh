#!/bin/bash
# Build the experimental Metal backend and optionally run synthetic validation.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build-metal"
jobs="${BUILD_JOBS:-2}"
validate=0

usage() {
  cat <<'EOF'
Usage: tools/build-metal-macos.sh [--validate] [--jobs N]

Environment:
  CMAKE_BIN      CMake executable (default: cmake)
  OPENMP_PREFIX  Prefix containing include/omp.h and lib/libomp.dylib
                 (default: brew --prefix libomp)
  PYTHON_BIN     Python executable for --validate (default: python3)
  BUILD_JOBS     Parallel build jobs (default: 2)

The build writes lib_lightgbm.dylib in the repository root and links it into
python-package/. --validate runs five small synthetic model checks and writes
build-metal/metal_validation.json. It does not run a performance benchmark.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --validate) validate=1; shift ;;
    --jobs)
      [[ $# -ge 2 ]] || { echo "--jobs requires a value" >&2; exit 2; }
      jobs="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be a positive integer" >&2; exit 2; }
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
  echo "Metal training requires Apple Silicon macOS." >&2
  exit 2
}

cmake_bin="${CMAKE_BIN:-cmake}"
command -v "$cmake_bin" >/dev/null 2>&1 || { echo "CMake not found: $cmake_bin" >&2; exit 2; }
if [[ -n "${OPENMP_PREFIX:-}" ]]; then
  openmp_prefix="$OPENMP_PREFIX"
elif command -v brew >/dev/null 2>&1; then
  openmp_prefix="$(brew --prefix libomp)"
else
  echo "Set OPENMP_PREFIX to an installed libomp prefix, or install Homebrew libomp." >&2
  exit 2
fi
[[ -f "$openmp_prefix/include/omp.h" && -f "$openmp_prefix/lib/libomp.dylib" ]] || {
  echo "libomp headers or library missing under: $openmp_prefix" >&2
  exit 2
}

"$cmake_bin" -S "$repo_root" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release -DUSE_METAL=ON -DUSE_OPENMP=ON \
  -DUSE_HOMEBREW_FALLBACK=OFF \
  "-DOpenMP_C_FLAGS=-Xpreprocessor -fopenmp -I$openmp_prefix/include" \
  "-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp -I$openmp_prefix/include" \
  -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp \
  "-DOpenMP_omp_LIBRARY=$openmp_prefix/lib/libomp.dylib"
"$cmake_bin" --build "$build_dir" -j "$jobs"

[[ -f "$repo_root/lib_lightgbm.dylib" ]] || { echo "Built library not found" >&2; exit 1; }
ln -sfn ../lib_lightgbm.dylib "$repo_root/python-package/lib_lightgbm.dylib"
echo "Built: $repo_root/lib_lightgbm.dylib"

if [[ "$validate" -eq 1 ]]; then
  python_bin="${PYTHON_BIN:-python3}"
  command -v "$python_bin" >/dev/null 2>&1 || { echo "Python not found: $python_bin" >&2; exit 2; }
  PYTHONPATH="$repo_root/python-package${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" "$repo_root/examples/python-guide/metal_validate.py" \
    --output "$build_dir/metal_validation.json"
fi

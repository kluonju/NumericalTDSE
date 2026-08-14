#!/usr/bin/env bash
# 1D harmonic oscillator, displaced Gaussian, RK4 (cheap demo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/build/bin/tdse"
if [[ ! -x "$BIN" ]]; then
    echo "Build first: cmake -S ${ROOT} -B ${ROOT}/build -DCMAKE_CXX_COMPILER=g++ && cmake --build ${ROOT}/build -j"
    exit 1
fi
exec "$BIN" "${ROOT}/examples/harmonic_1d.in"

#!/usr/bin/env bash
# 1D harmonic oscillator, displaced Gaussian, RK4 (cheap demo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/build/bin/tdse"
if [[ ! -x "$BIN" ]]; then
    echo "Build first: cmake -S ${ROOT} -B ${ROOT}/build && cmake --build ${ROOT}/build -j"
    exit 1
fi
exec "$BIN" \
    --dim 1 --electrons 1 --mode exact \
    --trap harmonic --omega 1 --x0 1 --alpha 1 \
    --propagator rk4 --dt 0.02 --T 0.40 \
    --prec 1e-4 --order 7 --L 8 \
    --no-ident-check \
    --output harmonic_1d.csv \
    --plot harmonic_1d

#!/usr/bin/env bash
# Hybrid MPI + OpenMP launch (2 ranks × OMP_NUM_THREADS cores).
# Exact N-body does not speed up with extra MPI ranks; use this for mode='orbital'.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/build/bin/tdse"
INPUT="${1:-${ROOT}/examples/orbitals_4e.in}"
RANKS="${RANKS:-2}"
THREADS="${OMP_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${THREADS}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-true}"
export OMP_PLACES="${OMP_PLACES:-cores}"
if [[ ! -x "$BIN" ]]; then
    echo "Build first: cmake -S ${ROOT} -B ${ROOT}/build -DCMAKE_CXX_COMPILER=g++ -DNUMTDSE_MPI=ON && cmake --build ${ROOT}/build -j"
    exit 1
fi
exec mpirun --bind-to none -np "${RANKS}" "$BIN" "${INPUT}"

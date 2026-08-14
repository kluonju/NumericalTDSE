#!/usr/bin/env bash
# Run a namelist job, then plot observables (and 1D wave functions if present).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/build/bin/tdse"
IN="${1:-${ROOT}/examples/harmonic_1d.in}"
if [[ ! -x "$BIN" ]]; then
    echo "Build first: cmake -S ${ROOT} -B ${ROOT}/build -DCMAKE_CXX_COMPILER=g++ && cmake --build ${ROOT}/build -j --target tdse"
    exit 1
fi
"$BIN" "$IN"

# Infer the CSV name from prefix= or output= in the namelist (last assignment wins).
PREFIX="$(awk -F= '
  BEGIN { IGNORECASE=1 }
  $0 ~ /^[[:space:]]*prefix[[:space:]]*=/ {
    gsub(/['\''"]/, "", $2); gsub(/,.*/, "", $2); gsub(/[[:space:]]/, "", $2); p=$2
  }
  $0 ~ /^[[:space:]]*output[[:space:]]*=/ {
    gsub(/['\''"]/, "", $2); gsub(/,.*/, "", $2); gsub(/[[:space:]]/, "", $2); o=$2
  }
  END { if (o != "") print o; else if (p != "") print p "_observables.csv"; else print "observables.csv" }
' "$IN")"
CSV="$PREFIX"
if [[ ! -f "$CSV" ]]; then
    CSV="$(basename "$CSV")"
fi
if [[ -f "$CSV" ]]; then
    python3 "${ROOT}/examples/compare_analytic.py" "$CSV" || true
    python3 "${ROOT}/examples/plot_observables.py" "$CSV" || echo "skip plot_observables (pip install matplotlib)"
fi

PLOT="$(awk -F= '
  BEGIN { IGNORECASE=1 }
  $0 ~ /^[[:space:]]*plot[[:space:]]*=/ {
    gsub(/['\''"]/, "", $2); gsub(/,.*/, "", $2); gsub(/[[:space:]]/, "", $2); print $2
  }
' "$IN" | tail -1)"
if [[ -n "${PLOT}" ]]; then
    for tag in t0 tT n0 n1 n2 n3 n4 n5 n6 n7; do
        if [[ -f "${PLOT}_${tag}_re.line" || -f "${PLOT}_${tag}_re" ]]; then
            extra=()
            if [[ "$tag" == n* ]]; then
                extra=(--analytic hoeig --n "${tag#n}")
            fi
            python3 "${ROOT}/examples/plot_wavefunction.py" "${PLOT}_${tag}" \
                -o "${PLOT}_${tag}_psi.png" "${extra[@]+"${extra[@]}"}" || true
        fi
    done
fi

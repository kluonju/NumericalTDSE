#!/usr/bin/env bash
# 1D harmonic oscillator vs coherent-state analytic (RK4 demo) + plots.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "${ROOT}/examples/run_and_plot.sh" "${ROOT}/examples/harmonic_1d.in"

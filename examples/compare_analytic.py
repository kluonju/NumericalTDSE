#!/usr/bin/env python3
"""Print RMS / max errors of numerical CSV vs analytic columns (no plotting).

Usage:
  python3 examples/compare_analytic.py harmonic_1d_observables.csv
  python3 examples/compare_analytic.py run.csv --tol-dipole 1e-3 --tol-energy 1e-3
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def _f(x: str) -> float:
    x = x.strip().lower()
    if x in ("nan", "+nan", "-nan", "inf", "+inf", "-inf", ""):
        return float("nan")
    return float(x)


def rms(vals):
    if not vals:
        return float("nan")
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_file")
    ap.add_argument("--tol-dipole", type=float, default=None)
    ap.add_argument("--tol-energy", type=float, default=None)
    ap.add_argument("--tol-overlap", type=float, default=None, help="minimum allowed overlap")
    args = ap.parse_args()

    path = Path(args.csv_file)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"empty CSV: {path}", file=sys.stderr)
        return 2

    dmu, dE, ov = [], [], []
    for r in rows:
        mu = _f(r["dipole"])
        E = _f(r["energy"])
        mua = _f(r.get("dipole_analytic", "nan"))
        Ea = _f(r.get("energy_analytic", "nan"))
        o = _f(r.get("overlap_analytic", "0"))
        if math.isfinite(mua):
            dmu.append(abs(mu - mua))
        if math.isfinite(Ea):
            dE.append(abs(E - Ea))
        if o > 0.0:
            ov.append(o)

    print(f"file: {path}")
    print(f"  steps: {len(rows)}")
    status = 0
    if dmu:
        print(f"  |μ − μ_ana|  max={max(dmu):.3e}  rms={rms(dmu):.3e}")
        if args.tol_dipole is not None and max(dmu) > args.tol_dipole:
            print(f"  FAIL dipole max error > {args.tol_dipole}", file=sys.stderr)
            status = 1
    else:
        print("  no analytic dipole column")
    if dE:
        print(f"  |E − E_ana|  max={max(dE):.3e}  rms={rms(dE):.3e}")
        if args.tol_energy is not None and max(dE) > args.tol_energy:
            print(f"  FAIL energy max error > {args.tol_energy}", file=sys.stderr)
            status = 1
    else:
        print("  no analytic energy column")
    if ov:
        print(f"  |⟨num|ana⟩|  min={min(ov):.12f}  last={ov[-1]:.12f}")
        if args.tol_overlap is not None and min(ov) < args.tol_overlap:
            print(f"  FAIL overlap min < {args.tol_overlap}", file=sys.stderr)
            status = 1
    else:
        print("  no wave-function overlap (enable validate_free / validate_ho)")
    return status


if __name__ == "__main__":
    sys.exit(main())

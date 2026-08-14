#!/usr/bin/env python3
"""Print RMS / max errors of numerical CSV vs analytic columns (no plotting).

Usage:
  python3 examples/compare_analytic.py harmonic_1d_observables.csv
  python3 examples/compare_analytic.py eigen_1d_observables.csv --tol-energy 1e-3
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
    ap.add_argument("--tol-residual", type=float, default=None, help="maximum ||(H-E)ψ||")
    args = ap.parse_args()

    path = Path(args.csv_file)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"empty CSV: {path}", file=sys.stderr)
        return 2

    fields = rows[0].keys()
    if "state" in fields and "energy" in fields:
        return compare_spectrum(path, rows, args)
    return compare_tdse(path, rows, args)


def compare_spectrum(path, rows, args) -> int:
    dE, ov, res = [], [], []
    print(f"file: {path}")
    print(f"  states: {len(rows)}")
    for r in rows:
        n = int(float(r["state"]))
        E = _f(r["energy"])
        Ea = _f(r.get("energy_analytic", "nan"))
        o = _f(r.get("overlap_analytic", "0"))
        rs = _f(r.get("residual", "nan"))
        line = f"  n={n}  E={E:.10f}"
        if math.isfinite(Ea):
            dE.append(abs(E - Ea))
            line += f"  E_ana={Ea:.10f}  |ΔE|={abs(E - Ea):.3e}"
        if o > 0.0:
            ov.append(o)
            line += f"  |⟨n|ana⟩|={o:.10f}"
        if math.isfinite(rs):
            res.append(rs)
            line += f"  residual={rs:.3e}"
        print(line)
    status = 0
    if dE:
        print(f"  |E − E_ana|  max={max(dE):.3e}  rms={rms(dE):.3e}")
        if args.tol_energy is not None and max(dE) > args.tol_energy:
            print(f"  FAIL energy max error > {args.tol_energy}", file=sys.stderr)
            status = 1
    if ov:
        print(f"  |⟨num|ana⟩|  min={min(ov):.12f}")
        if args.tol_overlap is not None and min(ov) < args.tol_overlap:
            print(f"  FAIL overlap min < {args.tol_overlap}", file=sys.stderr)
            status = 1
    if res:
        print(f"  ||(H−E)ψ||  max={max(res):.3e}")
        if args.tol_residual is not None and max(res) > args.tol_residual:
            print(f"  FAIL residual max > {args.tol_residual}", file=sys.stderr)
            status = 1
    return status


def compare_tdse(path, rows, args) -> int:
    dmu, dE, ov, res = [], [], [], []
    for r in rows:
        mu = _f(r["dipole"])
        E = _f(r["energy"])
        mua = _f(r.get("dipole_analytic", "nan"))
        Ea = _f(r.get("energy_analytic", "nan"))
        o = _f(r.get("overlap_analytic", "0"))
        rs = _f(r.get("residual", "nan"))
        if math.isfinite(mua):
            dmu.append(abs(mu - mua))
        if math.isfinite(Ea):
            dE.append(abs(E - Ea))
        if o > 0.0:
            ov.append(o)
        if math.isfinite(rs):
            res.append(rs)

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
        print("  no wave-function overlap (enable validate_free / validate_ho, or a HO eigen job)")
    if res:
        print(f"  ||(H−E)ψ||  last={res[-1]:.3e}  min={min(res):.3e}")
        if args.tol_residual is not None and res[-1] > args.tol_residual:
            print(f"  FAIL residual last > {args.tol_residual}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())

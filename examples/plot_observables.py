#!/usr/bin/env python3
"""Plot NumericalTDSE observable CSV vs closed-form columns when present.

Usage:
  python3 examples/plot_observables.py harmonic_1d_observables.csv
  python3 examples/plot_observables.py run.csv -o obs.png --title '1D HO'
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def _f(x: str) -> float:
    x = x.strip()
    if x.lower() in ("nan", "+nan", "-nan", "inf", "+inf", "-inf", ""):
        return float("nan")
    return float(x)


def load_csv(path: Path) -> dict:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty CSV: {path}")
    cols = {k: [_f(r[k]) for r in rows] for k in rows[0]}
    return cols


def has_finite(xs) -> bool:
    return any(math.isfinite(v) for v in xs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_file")
    ap.add_argument("-o", "--output", help="image file (png/pdf/svg). Default: <csv>.png")
    ap.add_argument("--title", default="")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        return 2

    path = Path(args.csv_file)
    d = load_csv(path)
    t = d["t"]
    out = Path(args.output) if args.output else path.with_suffix(".png")

    nplot = 3
    if has_finite(d.get("overlap_analytic", [])) and max(d["overlap_analytic"]) > 0:
        nplot = 4
    fig, axes = plt.subplots(nplot, 1, figsize=(7.2, 2.15 * nplot), sharex=True)
    if nplot == 1:
        axes = [axes]

    axes[0].plot(t, d["dipole"], "o-", ms=3, label="numerical")
    if has_finite(d.get("dipole_analytic", [])):
        axes[0].plot(t, d["dipole_analytic"], "--", label="analytic")
    axes[0].set_ylabel(r"dipole $\mu$")
    axes[0].legend(frameon=False)

    axes[1].plot(t, d["energy"], "o-", ms=3, label="numerical")
    if has_finite(d.get("energy_analytic", [])):
        axes[1].plot(t, d["energy_analytic"], "--", label="analytic")
    axes[1].set_ylabel("energy $E$")
    axes[1].legend(frameon=False)

    axes[2].plot(t, d["norm"], "o-", ms=3)
    axes[2].set_ylabel(r"$\|\psi\|$")

    if nplot == 4:
        axes[3].plot(t, d["overlap_analytic"], "o-", ms=3)
        axes[3].set_ylabel(r"$|\langle\mathrm{num}|\mathrm{ana}\rangle|$")
        axes[3].set_ylim(0.999, 1.0001)

    axes[-1].set_xlabel("$t$ (a.u.)")
    title = args.title or path.stem
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())

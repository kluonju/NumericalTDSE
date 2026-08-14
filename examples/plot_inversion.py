#!/usr/bin/env python3
"""Plot TGK08 inversion history and recovered potentials.

Usage:
  python3 examples/plot_inversion.py invert_2e1d_observables.csv
  python3 examples/plot_inversion.py invert_2e2d_observables.csv --density invert_2e2d_density.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


def read_csv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot density-to-potential inversion results")
    ap.add_argument("history", help="inversion history CSV (iter,l1,...)")
    ap.add_argument("--density", help="density/potential table CSV (optional)")
    ap.add_argument("-o", "--output", help="output figure path")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("plot_inversion.py needs matplotlib (pip install matplotlib)", file=sys.stderr)
        return 1

    _, hist = read_csv(args.history)
    if not hist:
        print("empty history", file=sys.stderr)
        return 1

    density_path = args.density
    if density_path is None:
        stem = args.history
        for suffix in ("_observables.csv", ".csv"):
            if stem.endswith(suffix):
                cand = stem[: -len(suffix)] + "_density.csv"
                if os.path.isfile(cand):
                    density_path = cand
                    break

    iters = [int(float(r["iter"])) for r in hist]
    l1 = [float(r["l1"]) for r in hist]

    dens_fields: list[str] = []
    dens_rows: list[dict[str, str]] = []
    if density_path and os.path.isfile(density_path):
        dens_fields, dens_rows = read_csv(density_path)

    is_2d = dens_rows and "y" in dens_fields
    ncols = 2 if dens_rows else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.2), squeeze=False)
    ax0 = axes[0][0]
    ax0.semilogy(iters, l1, "o-", color="#1f4e79")
    ax0.set_xlabel("iteration")
    ax0.set_ylabel(r"$\int |n-n^*|\,dr$")
    ax0.set_title("TGK08 density residual")
    ax0.grid(True, alpha=0.3)

    if dens_rows:
        ax1 = axes[0][1]
        if is_2d:
            xs = sorted({float(r["x"]) for r in dens_rows})
            mid = xs[len(xs) // 2]
            line = [r for r in dens_rows if abs(float(r["x"]) - mid) < 1e-12]
            line.sort(key=lambda r: float(r["y"]))
            y = [float(r["y"]) for r in line]
            ax1.plot(y, [float(r["v_true"]) for r in line], "k-", label=r"$v_{\mathrm{true}}(0,y)$")
            ax1.plot(y, [float(r["v_inv"]) for r in line], "C0--", label=r"$v_{\mathrm{inv}}$")
            ax1.plot(y, [float(r["v_s"]) for r in line], "C1:", label=r"$v_s$")
            ax1.set_xlabel("y")
            ax1.set_title("cut at x = 0")
        else:
            x = [float(r["x"]) for r in dens_rows]
            ax1.plot(x, [float(r["v_true"]) for r in dens_rows], "k-", label=r"$v_{\mathrm{true}}$")
            ax1.plot(x, [float(r["v_inv"]) for r in dens_rows], "C0--", label=r"$v_{\mathrm{inv}}$")
            ax1.plot(x, [float(r["v_s"]) for r in dens_rows], "C1:", label=r"$v_s$ (KS)")
            ax1.set_xlabel("x")
            ax1.set_title("recovered potentials")
        ax1.set_ylabel("potential (a.u.)")
        ax1.legend(frameon=False)
        ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    out = args.output
    if not out:
        out = os.path.splitext(args.history)[0] + "_invert.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

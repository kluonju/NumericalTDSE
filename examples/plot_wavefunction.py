#!/usr/bin/env python3
"""Plot MRCPP 1D line dumps (Re ψ, Im ψ) and optional analytic |ψ|.

MRCPP linePlot files are two columns: x  value   (optional .line suffix).

Usage:
  python3 examples/plot_wavefunction.py ho_t0_re.line ho_t0_im.line \\
      --analytic ho --x0 1 --omega 1 --t 0 -o psi_t0.png
  python3 examples/plot_wavefunction.py free_tT_re.line free_tT_im.line \\
      --analytic free --alpha 1 --x0 0 --k0 0 --t 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analytic_ref import free_gaussian, ho_coherent


def load_line(path: Path):
    xs, ys = [], []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("!"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 2:
                continue
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
    if not xs:
        raise SystemExit(f"no data in {path}")
    return xs, ys


def find_pair(prefix: str):
    """Accept a prefix such as harmonic_1d_t0 and look for _re/_im files."""
    cands = [
        (Path(prefix + "_re.line"), Path(prefix + "_im.line")),
        (Path(prefix + "_re"), Path(prefix + "_im")),
        (Path(prefix + ".re"), Path(prefix + ".im")),
    ]
    for a, b in cands:
        if a.is_file() and b.is_file():
            return a, b
    raise SystemExit(f"could not find Re/Im line files for prefix {prefix!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="re_file im_file, or a common prefix")
    ap.add_argument("--analytic", choices=("none", "free", "ho"), default="none")
    ap.add_argument("--x0", type=float, default=1.0)
    ap.add_argument("--k0", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("-o", "--output", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        return 2

    if len(args.files) == 1:
        re_path, im_path = find_pair(args.files[0])
    elif len(args.files) >= 2:
        re_path, im_path = Path(args.files[0]), Path(args.files[1])
    else:
        ap.error("give Re and Im line files, or a prefix")

    x, re = load_line(re_path)
    xi, im = load_line(im_path)
    if len(x) != len(xi):
        print("warning: Re/Im grids differ in length; interpolating onto Re grid", file=sys.stderr)
        im = [im[min(i, len(im) - 1)] for i in range(len(x))]
    dens = [a * a + b * b for a, b in zip(re, im)]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    axes[0].plot(x, re, label=r"Re $\psi$")
    axes[0].plot(x, im, label=r"Im $\psi$")
    axes[0].set_ylabel(r"$\psi(x)$")
    axes[0].legend(frameon=False)

    axes[1].plot(x, dens, label=r"$|\psi|^2$ numerical")
    if args.analytic == "free":
        ana = [abs(free_gaussian(xi, args.x0, args.alpha, args.t, args.k0)) ** 2 for xi in x]
        axes[1].plot(x, ana, "--", label=r"$|\psi|^2$ analytic")
    elif args.analytic == "ho":
        ana = [abs(ho_coherent(xi, args.x0, args.k0, args.omega, args.t)) ** 2 for xi in x]
        axes[1].plot(x, ana, "--", label=r"$|\psi|^2$ analytic")
    axes[1].set_ylabel(r"$|\psi|^2$")
    axes[1].set_xlabel("$x$ (a.u.)")
    axes[1].legend(frameon=False)

    fig.suptitle(args.title or re_path.stem.replace("_re", ""))
    fig.tight_layout()
    out = Path(args.output) if args.output else re_path.with_name(re_path.stem.replace("_re", "") + "_psi.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())

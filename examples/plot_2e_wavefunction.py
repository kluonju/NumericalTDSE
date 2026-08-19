#!/usr/bin/env python3
"""Plot exact 1D two-electron MRCPP surface dumps ψ(x1, x2).

MRCPP surfPlot files are:  x1  x2  value   (optional .surf suffix).

Usage:
  python3 examples/plot_2e_wavefunction.py ho_2e_singlet_n0
  python3 examples/plot_2e_wavefunction.py ho_2e_singlet_n0 ho_2e_triplet_n0 \\
      --omega 1 -o ho_2e_spin_gs.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analytic_ref import ho_eigen


def load_surf(path: Path):
    xs, ys, vs = [], [], []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("!"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 3:
                continue
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
            vs.append(float(parts[2]))
    if not xs:
        raise SystemExit(f"no data in {path}")
    x = np.array(xs)
    y = np.array(ys)
    z = np.array(vs)
    ux = np.unique(np.round(x, 10))
    uy = np.unique(np.round(y, 10))
    nx, ny = ux.size, uy.size
    if nx * ny != z.size:
        raise SystemExit(f"{path}: {z.size} points, but {nx}×{ny} unique axes")
    # MRCPP walks A (x1) inner, then B (x2).
    if abs(x[1] - x[0]) > 1e-14:
        # x1 varies fastest: already (x2, x1) for imshow
        z = z.reshape(ny, nx)
    else:
        # x2 varies fastest: (x1, x2) → (x2, x1)
        z = z.reshape(nx, ny).T
    return ux, uy, z


def find_pair(prefix: str):
    cands = [
        (Path(prefix + "_re.surf"), Path(prefix + "_im.surf")),
        (Path(prefix + "_re"), Path(prefix + "_im")),
    ]
    for a, b in cands:
        if a.is_file() and b.is_file():
            return a, b
    raise SystemExit(f"could not find Re/Im .surf files for prefix {prefix!r}")


def analytic_psi(X, Y, kind: str, omega: float):
    xx, yy = np.meshgrid(X, Y)
    if kind == "singlet":
        p0x = np.vectorize(lambda t: ho_eigen(float(t), 0, omega).real)(xx)
        p0y = np.vectorize(lambda t: ho_eigen(float(t), 0, omega).real)(yy)
        return p0x * p0y
    p0x = np.vectorize(lambda t: ho_eigen(float(t), 0, omega).real)(xx)
    p1x = np.vectorize(lambda t: ho_eigen(float(t), 1, omega).real)(xx)
    p0y = np.vectorize(lambda t: ho_eigen(float(t), 0, omega).real)(yy)
    p1y = np.vectorize(lambda t: ho_eigen(float(t), 1, omega).real)(yy)
    return (p0x * p1y - p1x * p0y) / math.sqrt(2.0)


def infer_kind(prefix: str) -> str:
    s = prefix.lower()
    if "triplet" in s:
        return "triplet"
    if "singlet" in s:
        return "singlet"
    return ""


def plot_one(ax_re, ax_n, X, Y, re, im, title: str, kind: str, omega: float):
    dens = re * re + im * im
    vmax = float(np.max(np.abs(re))) or 1.0
    nmax = float(np.max(dens)) or 1.0
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    im0 = ax_re.imshow(
        re,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
    )
    ax_re.plot([X.min(), X.max()], [Y.min(), Y.max()], "k--", lw=0.6, alpha=0.4)
    ax_re.set_title(title + r"  Re $\psi(x_1,x_2)$")
    ax_re.set_xlabel(r"$x_1$ (a.u.)")
    ax_re.set_ylabel(r"$x_2$ (a.u.)")
    im1 = ax_n.imshow(
        dens,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=nmax,
        aspect="equal",
    )
    ax_n.plot([X.min(), X.max()], [Y.min(), Y.max()], "w--", lw=0.6, alpha=0.5)
    ax_n.set_title(title + r"  $|\psi|^2$")
    ax_n.set_xlabel(r"$x_1$ (a.u.)")
    ax_n.set_ylabel(r"$x_2$ (a.u.)")
    if kind:
        ana = analytic_psi(X, Y, kind, omega)
        if float(np.sum(ana * re)) < 0.0:
            ana = -ana
        ax_re.contour(X, Y, ana, levels=6, colors="k", linewidths=0.4, alpha=0.55)
    return im0, im1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prefixes", nargs="+", help="plot prefix(es), e.g. ho_2e_singlet_n0")
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("-o", "--output", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        return 2

    n = len(args.prefixes)
    fig, axes = plt.subplots(n, 2, figsize=(8.8, 4.0 * n), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])

    for i, prefix in enumerate(args.prefixes):
        re_path, im_path = find_pair(prefix)
        X, Y, re = load_surf(re_path)
        _Xi, _Yi, im = load_surf(im_path)
        if re.shape != im.shape:
            print("warning: Re/Im grids differ; using Re grid for |ψ|²", file=sys.stderr)
            im = np.zeros_like(re)
        kind = infer_kind(prefix)
        label = Path(prefix).name.replace("_n0", "")
        im0, im1 = plot_one(axes[i, 0], axes[i, 1], X, Y, re, im, label, kind, args.omega)
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

    fig.suptitle(args.title or "Exact 1D two-electron ground states")
    out = Path(args.output) if args.output else Path(args.prefixes[0] + "_psi.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Closed-form references matching include/tdse/analytic.hpp (atomic units)."""

from __future__ import annotations

import cmath
import math
from typing import Optional

PI = math.pi


def free_gaussian(x: float, x0: float, alpha: float, t: float, k0: float = 0.0) -> complex:
    den = 1.0 + 1j * alpha * t
    nrm = (alpha / PI) ** 0.25
    pref = nrm / cmath.sqrt(den)
    dx = x - x0 - k0 * t
    boost = 1j * k0 * x - 1j * 0.5 * k0 * k0 * t
    return pref * cmath.exp(boost - 0.5 * alpha * dx * dx / den)


def ho_coherent(x: float, x0: float, k0: float, omega: float, t: float) -> complex:
    xc = x0 * math.cos(omega * t) + (k0 / omega) * math.sin(omega * t)
    pc = k0 * math.cos(omega * t) - omega * x0 * math.sin(omega * t)
    nrm = (omega / PI) ** 0.25
    phase = 1j * pc * (x - 0.5 * xc) - 1j * 0.5 * omega * t
    return nrm * cmath.exp(-0.5 * omega * (x - xc) ** 2 + phase)


def analytic_dipole(
    trap: str,
    t: float,
    x0: float,
    k0: float,
    omega: float = 1.0,
    E0: float = 0.0,
    omega_L: float = 0.5,
    envelope: bool = False,
) -> Optional[float]:
    trap = trap.lower()
    if trap in ("free", "none"):
        return x0 + k0 * t
    if trap not in ("harmonic", "ho"):
        return None
    if E0 == 0.0 or envelope:
        return x0 * math.cos(omega * t) + (k0 / omega) * math.sin(omega * t)
    den = omega * omega - omega_L * omega_L
    if abs(den) <= 1e-14:
        return None
    A = x0
    B = (k0 - E0 * omega_L / den) / omega
    return A * math.cos(omega * t) + B * math.sin(omega * t) + E0 * math.sin(omega_L * t) / den


def analytic_energy(
    trap: str,
    alpha: float,
    x0: float,
    k0: float = 0.0,
    omega: float = 1.0,
    dim: int = 1,
    E0: float = 0.0,
) -> Optional[float]:
    if E0 != 0.0:
        return None
    boost = 0.5 * k0 * k0
    trap = trap.lower()
    if trap in ("free", "none"):
        return 0.25 * alpha * dim + boost
    if trap in ("harmonic", "ho") and alpha > 0.0:
        width = 0.25 * dim * (alpha + omega * omega / alpha)
        return width + 0.5 * omega * omega * x0 * x0 + boost
    return None


if __name__ == "__main__":
    e_ho = analytic_energy("harmonic", alpha=1.0, x0=1.0, k0=0.0, omega=1.0, dim=1)
    e_sm = analytic_energy("harmonic", alpha=1.0, x0=0.5, k0=0.0, omega=1.0, dim=1)
    e_fr = analytic_energy("free", alpha=1.0, x0=0.0, k0=0.0, dim=1)
    assert e_ho is not None and abs(e_ho - 1.0) < 1e-12, e_ho
    assert e_sm is not None and abs(e_sm - 0.625) < 1e-12, e_sm
    assert e_fr is not None and abs(e_fr - 0.25) < 1e-12, e_fr
    d0 = analytic_dipole("harmonic", 0.0, 1.0, 0.0, 1.0)
    assert d0 is not None and abs(d0 - 1.0) < 1e-12
    print("analytic_ref self-check ok")

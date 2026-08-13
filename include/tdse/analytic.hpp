#pragma once

/**
 * Analytic initial data and potentials as MRCPP RepresentableFunction objects.
 *
 * MRCPP projects any class that implements
 *     double evalf(const mrcpp::Coord<D> &r) const
 * onto an adaptive FunctionTree. Time-dependent potentials expose set_time()
 * so the same object can be re-projected at each step.
 *
 * Coordinate layout
 * -----------------
 * Exact N-body, 1D electrons (D = N):
 *     r[i] is the position of electron i = 0..N-1.
 * One electron in D spatial dimensions:
 *     r[0..D-1] is the Cartesian coordinate of that electron.
 */

#include "tdse/parameters.hpp"

#include "MRCPP/MWFunctions"

#include <array>
#include <cmath>
#include <complex>
#include <limits>

namespace tdse {

inline constexpr double PI = 3.14159265358979323846;

/** Harmonic-oscillator / Gaussian envelope (real part of the initial orbital). */
template <int D>
class InitialWavefunctionReal : public mrcpp::RepresentableFunction<D> {
public:
    explicit InitialWavefunctionReal(const Parameters &p, double center_x = std::numeric_limits<double>::quiet_NaN())
            : p_(p)
            , center_(std::isnan(center_x) ? p.x0 : center_x) {}

    double evalf(const mrcpp::Coord<D> &r) const override {
        const double g = gaussian_envelope(r);
        const double phase = boost_phase(r);
        return g * std::cos(phase);
    }

protected:
    Parameters p_;
    double center_{0.0};

    double gaussian_envelope(const mrcpp::Coord<D> &r) const {
        const int n = p_.n_electrons;
        const bool nbody_1d = (p_.representation == Representation::Exact && p_.spatial_dim == 1 && n > 1);

        if (nbody_1d) {
            // Product of 1D Gaussians, optionally antisymmetrised for N = 2.
            const double nrm1 = std::pow(p_.alpha / PI, 0.25);
            auto orb = [&](double x, double center) {
                return nrm1 * std::exp(-0.5 * p_.alpha * (x - center) * (x - center));
            };
            if (n == 2 && p_.fermion) {
                const double a = orb(r[0], p_.x0) * orb(r[1], -p_.x0);
                const double b = orb(r[0], -p_.x0) * orb(r[1], p_.x0);
                return (a - b) / std::sqrt(2.0);
            }
            double val = 1.0;
            const double nrm = std::pow(nrm1, static_cast<double>(D));
            for (int i = 0; i < D; ++i) {
                const double c = (i == 0) ? center_ : 0.0;
                val *= std::exp(-0.5 * p_.alpha * (r[i] - c) * (r[i] - c));
            }
            return nrm * val;
        }

        // Single particle in D dimensions, displaced along x.
        const double nrm = std::pow(p_.alpha / PI, 0.25 * D);
        double r2 = 0.0;
        for (int d = 0; d < D; ++d) {
            const double x = r[d] - ((d == 0) ? center_ : 0.0);
            r2 += x * x;
        }
        return nrm * std::exp(-0.5 * p_.alpha * r2);
    }

    double boost_phase(const mrcpp::Coord<D> &r) const {
        return p_.k0 * r[0];
    }
};

/** Imaginary part of the same boosted Gaussian (sin of the plane-wave phase). */
template <int D>
class InitialWavefunctionImag : public InitialWavefunctionReal<D> {
public:
    explicit InitialWavefunctionImag(const Parameters &p, double center_x = std::numeric_limits<double>::quiet_NaN())
            : InitialWavefunctionReal<D>(p, center_x) {}

    double evalf(const mrcpp::Coord<D> &r) const override {
        const double g = this->gaussian_envelope(r);
        const double phase = this->boost_phase(r);
        return g * std::sin(phase);
    }
};

/**
 * Time-dependent potential V(r, t).
 *
 * Atomic units, standard molecular Hamiltonian:
 *   V_trap  : harmonic, free, or soft Coulomb nuclear well
 *   V_laser : −E(t) · μ, dipole approximation, μ = Σ x_i
 *   V_ee    : soft Coulomb between electrons (exact N-body only)
 */
template <int D>
class TimeDependentPotential : public mrcpp::RepresentableFunction<D> {
public:
    explicit TimeDependentPotential(const Parameters &p)
            : p_(p)
            , t_(0.0) {}

    void set_time(double t) { t_ = t; }
    double time() const { return t_; }

    double evalf(const mrcpp::Coord<D> &r) const override { return eval_at(r, t_); }

    double eval_at(const mrcpp::Coord<D> &r, double t) const {
        return trap(r) + laser(r, t) + electron_electron(r);
    }

    /** Laser electric field E(t) along x (atomic units). */
    double electric_field(double t) const {
        if (p_.E0 == 0.0) {
            return 0.0;
        }
        double env = 1.0;
        if (p_.laser_envelope) {
            if (t <= 0.0 || t >= p_.T) {
                return 0.0;
            }
            const double s = std::sin(PI * t / p_.T);
            env = s * s;
        }
        return p_.E0 * env * std::sin(p_.omega_L * t);
    }

private:
    Parameters p_;
    double t_;

    double trap(const mrcpp::Coord<D> &r) const {
        switch (p_.trap) {
            case TrapKind::None:
                return 0.0;
            case TrapKind::Harmonic: {
                double r2 = 0.0;
                for (int d = 0; d < D; ++d) {
                    r2 += r[d] * r[d];
                }
                return 0.5 * p_.omega * p_.omega * r2;
            }
            case TrapKind::SoftAtom: {
                const bool nbody_1d = (p_.representation == Representation::Exact && p_.spatial_dim == 1 && p_.n_electrons > 1);
                if (nbody_1d) {
                    double v = 0.0;
                    for (int i = 0; i < D; ++i) {
                        v += -p_.Z / std::sqrt(r[i] * r[i] + p_.soft_a * p_.soft_a);
                    }
                    return v;
                }
                double r2 = 0.0;
                for (int d = 0; d < D; ++d) {
                    r2 += r[d] * r[d];
                }
                return -p_.Z / std::sqrt(r2 + p_.soft_a * p_.soft_a);
            }
        }
        return 0.0;
    }

    double laser(const mrcpp::Coord<D> &r, double t) const {
        const double E = electric_field(t);
        if (E == 0.0) {
            return 0.0;
        }
        const bool nbody_1d = (p_.representation == Representation::Exact && p_.spatial_dim == 1 && p_.n_electrons > 1);
        if (nbody_1d) {
            double dipole = 0.0;
            for (int i = 0; i < D; ++i) {
                dipole += r[i];
            }
            return -E * dipole;
        }
        return -E * r[0];
    }

    double electron_electron(const mrcpp::Coord<D> &r) const {
        const bool nbody_1d = (p_.representation == Representation::Exact && p_.spatial_dim == 1 && p_.n_electrons > 1);
        if (!nbody_1d) {
            return 0.0;
        }
        double v = 0.0;
        const double a2 = p_.soft_a * p_.soft_a;
        for (int i = 0; i < D; ++i) {
            for (int j = i + 1; j < D; ++j) {
                const double dx = r[i] - r[j];
                v += 1.0 / std::sqrt(dx * dx + a2);
            }
        }
        return v;
    }
};

/** Analytic free Gaussian for i ∂t ψ = −½ ∂²_x ψ (atomic units, 1D). */
inline std::complex<double> free_gaussian_1d(double x, double x0, double alpha, double t) {
    // ψ(x,0) = (α/π)^{1/4} exp(−α (x−x0)² / 2)
    // ψ(x,t) = (α/π)^{1/4} (1 + i α t)^{−1/2} exp(−α (x−x0)² / 2 / (1 + i α t))
    const std::complex<double> I(0.0, 1.0);
    const std::complex<double> den = 1.0 + I * alpha * t;
    const double nrm = std::pow(alpha / PI, 0.25);
    const std::complex<double> pref = nrm / std::sqrt(den);
    const double dx = x - x0;
    return pref * std::exp(-0.5 * alpha * dx * dx / den);
}

/** Coordinate function x_d, used to build the dipole density. */
template <int D>
class CoordinateFunction : public mrcpp::RepresentableFunction<D> {
public:
    explicit CoordinateFunction(int axis)
            : axis_(axis) {}

    double evalf(const mrcpp::Coord<D> &r) const override { return r[axis_]; }

private:
    int axis_;
};

} // namespace tdse

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
 * Exact N-body (D = n_e * spatial_dim ≤ 4):
 *     electron i occupies r[i*dim .. i*dim+dim-1].
 *     1D electrons (dim = 1): r[i] is the position of electron i.
 * One electron in D spatial dimensions:
 *     r[0..D-1] is the Cartesian coordinate of that electron.
 */

#include "tdse/parameters.hpp"

#include "MRCPP/MWFunctions"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>

namespace tdse {

inline constexpr double PI = 3.14159265358979323846;

inline bool is_exact_nbody(const Parameters &p) {
    return p.representation == Representation::Exact && p.n_electrons > 1;
}

inline std::complex<double> ho_eigen_1d(double x, int n, double omega);

/**
 * Exact two-electron spatial wave function with definite exchange parity.
 * Singlet: ψ(r1,r2)=+ψ(r2,r1). Triplet: ψ(r1,r2)=−ψ(r2,r1).
 * Electron i occupies r[i*dim .. i*dim+dim-1].
 */
template <int D>
double two_electron_spatial(const mrcpp::Coord<D> &r, const Parameters &p) {
    if constexpr (D != 2 && D != 4) {
        return 0.0;
    } else {
        const SpinKind spin = two_electron_spin(p);
        if (spin == SpinKind::Unspecified || p.n_electrons != 2) {
            return 0.0;
        }
        const int dim = p.spatial_dim;
        if (dim * 2 != D) {
            return 0.0;
        }

        auto gauss_orb = [&](int e, double cx) -> double {
            double r2 = 0.0;
            for (int a = 0; a < dim; ++a) {
                const double x = r[e * dim + a] - ((a == 0) ? cx : 0.0);
                r2 += x * x;
            }
            const double nrm = std::pow(p.alpha / PI, 0.25 * static_cast<double>(dim));
            return nrm * std::exp(-0.5 * p.alpha * r2);
        };
        auto ho_orb = [&](int e, int nx) -> double {
            double v = ho_eigen_1d(r[e * dim], nx, p.omega).real();
            for (int a = 1; a < dim; ++a) {
                v *= ho_eigen_1d(r[e * dim + a], 0, p.omega).real();
            }
            return v;
        };

        const bool use_ho = (p.trap == TrapKind::Harmonic && p.omega > 0.0 && std::abs(p.x0) < 1.0e-14);
        if (spin == SpinKind::Singlet) {
            if (use_ho) {
                return ho_orb(0, 0) * ho_orb(1, 0);
            }
            const double a = gauss_orb(0, p.x0) * gauss_orb(1, -p.x0);
            const double b = gauss_orb(0, -p.x0) * gauss_orb(1, p.x0);
            if (std::abs(p.x0) < 1.0e-14) {
                return a;
            }
            return (a + b) / std::sqrt(2.0);
        }

        if (use_ho) {
            return (ho_orb(0, 0) * ho_orb(1, 1) - ho_orb(0, 1) * ho_orb(1, 0)) / std::sqrt(2.0);
        }
        if (std::abs(p.x0) > 1.0e-14) {
            const double a = gauss_orb(0, p.x0) * gauss_orb(1, -p.x0);
            const double b = gauss_orb(0, -p.x0) * gauss_orb(1, p.x0);
            return (a - b) / std::sqrt(2.0);
        }
        const double prod = gauss_orb(0, 0.0) * gauss_orb(1, 0.0);
        return (r[0] - r[dim]) * prod;
    }
}

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
        const bool nbody_1d = is_exact_nbody(p_) && p_.spatial_dim == 1;

        if (n == 2 && is_exact_nbody(p_) && two_electron_spin(p_) != SpinKind::Unspecified) {
            return two_electron_spatial<D>(r, p_);
        }

        if (nbody_1d) {
            // Product of 1D Gaussians; N = 2 fermion without spin= is handled above.
            const double nrm1 = std::pow(p_.alpha / PI, 0.25);
            auto orb = [&](double x, double center) {
                return nrm1 * std::exp(-0.5 * p_.alpha * (x - center) * (x - center));
            };
            if constexpr (D >= 3) {
                if (n == 3 && p_.fermion) {
                    auto phi = [&](int k, double x) -> double {
                        if (p_.trap == TrapKind::Harmonic && p_.omega > 0.0) {
                            return ho_eigen_1d(x, k, p_.omega).real();
                        }
                        const double d = (std::abs(p_.x0) > 1.0e-14) ? p_.x0 : 0.7;
                        const double c = (k == 0) ? -d : (k == 1 ? 0.0 : d);
                        return orb(x, c);
                    };
                    double a[3][3];
                    for (int k = 0; k < 3; ++k) {
                        for (int e = 0; e < 3; ++e) {
                            a[k][e] = phi(k, r[e]);
                        }
                    }
                    const double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                                       a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                                       a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
                    return det / std::sqrt(6.0);
                }
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
                if (is_exact_nbody(p_)) {
                    const int dim = p_.spatial_dim;
                    double v = 0.0;
                    for (int i = 0; i < p_.n_electrons; ++i) {
                        double r2 = 0.0;
                        for (int d = 0; d < dim; ++d) {
                            const double x = r[i * dim + d];
                            r2 += x * x;
                        }
                        v += -p_.Z / std::sqrt(r2 + p_.soft_a * p_.soft_a);
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
        if (is_exact_nbody(p_)) {
            double dipole = 0.0;
            const int dim = p_.spatial_dim;
            for (int i = 0; i < p_.n_electrons; ++i) {
                dipole += r[i * dim];
            }
            return -E * dipole;
        }
        return -E * r[0];
    }

    double electron_electron(const mrcpp::Coord<D> &r) const {
        if (!p_.ee || !is_exact_nbody(p_)) {
            return 0.0;
        }
        double v = 0.0;
        const double a2 = p_.soft_a * p_.soft_a;
        const int dim = p_.spatial_dim;
        const int n = p_.n_electrons;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double r2 = 0.0;
                for (int d = 0; d < dim; ++d) {
                    const double dx = r[i * dim + d] - r[j * dim + d];
                    r2 += dx * dx;
                }
                v += 1.0 / std::sqrt(r2 + a2);
            }
        }
        return v;
    }
};

/** Analytic free Gaussian for i ∂t ψ = −½ ∂²_x ψ (atomic units, 1D), optional boost k0. */
inline std::complex<double> free_gaussian_1d(double x, double x0, double alpha, double t, double k0 = 0.0) {
    // ψ(x,0) = (α/π)^{1/4} exp(−α (x−x0)² / 2 + i k0 x)
    // ψ(x,t) = (α/π)^{1/4} (1 + i α t)^{−1/2}
    //          × exp(i k0 x − i k0² t / 2) exp(−α (x−x0−k0 t)² / 2 / (1 + i α t))
    const std::complex<double> I(0.0, 1.0);
    const std::complex<double> den = 1.0 + I * alpha * t;
    const double nrm = std::pow(alpha / PI, 0.25);
    const std::complex<double> pref = nrm / std::sqrt(den);
    const double dx = x - x0 - k0 * t;
    const std::complex<double> boost = I * k0 * x - I * 0.5 * k0 * k0 * t;
    return pref * std::exp(boost - 0.5 * alpha * dx * dx / den);
}

/**
 * 1D harmonic-oscillator coherent state (requires α = ω).
 * ⟨x⟩ = x0 cos(ω t) + (k0/ω) sin(ω t),  ⟨p⟩ = k0 cos(ω t) − ω x0 sin(ω t).
 */
inline std::complex<double> ho_coherent_1d(double x, double x0, double k0, double omega, double t) {
    const std::complex<double> I(0.0, 1.0);
    const double xc = x0 * std::cos(omega * t) + (k0 / omega) * std::sin(omega * t);
    const double pc = k0 * std::cos(omega * t) - omega * x0 * std::sin(omega * t);
    const double nrm = std::pow(omega / PI, 0.25);
    const std::complex<double> phase = I * pc * (x - 0.5 * xc) - I * 0.5 * omega * t;
    return nrm * std::exp(-0.5 * omega * (x - xc) * (x - xc) + phase);
}

/** Closed-form ⟨x⟩ for one electron (Ehrenfest). Driven HO assumes CW laser, no envelope. */
inline double analytic_dipole(const Parameters &p, double t) {
    if (p.representation != Representation::Exact || p.n_electrons != 1) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (p.trap == TrapKind::None) {
        return p.x0 + p.k0 * t;
    }
    if (p.trap != TrapKind::Harmonic || p.omega == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double w = p.omega;
    if (p.E0 == 0.0 || p.laser_envelope) {
        return p.x0 * std::cos(w * t) + (p.k0 / w) * std::sin(w * t);
    }
    const double om = p.omega_L;
    const double den = w * w - om * om;
    if (std::abs(den) <= 1.0e-14) {
        return std::numeric_limits<double>::quiet_NaN(); // resonant driving
    }
    const double A = p.x0;
    const double B = (p.k0 - p.E0 * om / den) / w;
    return A * std::cos(w * t) + B * std::sin(w * t) + p.E0 * std::sin(om * t) / den;
}

/** Closed-form energy of the initial Gaussian (conserved if E0 = 0). */
inline double analytic_energy(const Parameters &p) {
    if (p.E0 != 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (is_exact_nbody(p) && p.trap == TrapKind::Harmonic && p.omega > 0.0 && !p.ee) {
        const SpinKind spin = two_electron_spin(p);
        if (p.n_electrons == 2 && spin != SpinKind::Unspecified) {
            if (std::abs(p.k0) > 1.0e-14 || std::abs(p.x0) > 1.0e-14) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double e_singlet = p.omega * static_cast<double>(p.spatial_dim);
            return (spin == SpinKind::Triplet) ? (e_singlet + p.omega) : e_singlet;
        }
        if (p.fermion && p.spatial_dim == 1) {
            double E = 0.0;
            for (int k = 0; k < p.n_electrons; ++k) {
                E += p.omega * (static_cast<double>(k) + 0.5);
            }
            return E;
        }
        if (p.alpha > 0.0) {
            const double Dm = static_cast<double>(p.spatial_dim * p.n_electrons);
            const double width = 0.25 * Dm * (p.alpha + p.omega * p.omega / p.alpha);
            return width + 0.5 * p.omega * p.omega * p.x0 * p.x0 + 0.5 * p.k0 * p.k0;
        }
    }
    if (p.representation != Representation::Exact || p.n_electrons != 1) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double boost = 0.5 * p.k0 * p.k0;
    const double D = static_cast<double>(p.spatial_dim);
    if (p.trap == TrapKind::None) {
        return 0.25 * p.alpha * D + boost;
    }
    if (p.trap == TrapKind::Harmonic && p.alpha > 0.0) {
        const double width = 0.25 * D * (p.alpha + p.omega * p.omega / p.alpha);
        return width + 0.5 * p.omega * p.omega * p.x0 * p.x0 + boost;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

inline bool wants_analytic_overlap(const Parameters &p) {
    if (p.spatial_dim != 1 || p.n_electrons != 1 || p.representation != Representation::Exact) {
        return false;
    }
    if (p.validate_free) {
        return true;
    }
    if (p.validate_ho && p.E0 == 0.0 && p.alpha > 0.0 && std::abs(p.alpha - p.omega) < 1.0e-12) {
        return true;
    }
    return false;
}

inline int binom_int(int n, int k) {
    if (k < 0 || k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    if (k > n - k) {
        k = n - k;
    }
    long r = 1;
    for (int i = 1; i <= k; ++i) {
        r = r * (n - k + i) / i;
    }
    return static_cast<int>(r);
}

/** Shell index N = n1+…+nD of the k-th isotropic HO state (k is 0-based). */
inline int ho_cartesian_shell(int k, int dim) {
    if (dim <= 1) {
        return std::max(0, k);
    }
    int count = 0;
    for (int n = 0; n < 128; ++n) {
        const int deg = binom_int(n + dim - 1, dim - 1);
        if (count + deg > k) {
            return n;
        }
        count += deg;
    }
    return 128;
}

/** Physicists' Hermite polynomial H_n(x). */
inline double hermite_phys(int n, double x) {
    if (n <= 0) {
        return 1.0;
    }
    if (n == 1) {
        return 2.0 * x;
    }
    double Hnm2 = 1.0;
    double Hnm1 = 2.0 * x;
    double Hn = 0.0;
    for (int k = 2; k <= n; ++k) {
        Hn = 2.0 * x * Hnm1 - 2.0 * static_cast<double>(k - 1) * Hnm2;
        Hnm2 = Hnm1;
        Hnm1 = Hn;
    }
    return Hn;
}

/** 1D HO eigenfunction ψ_n(x) (real, phase convention of Abramowitz & Stegun). */
inline std::complex<double> ho_eigen_1d(double x, int n, double omega) {
    const double xi = std::sqrt(omega) * x;
    double fact = 1.0;
    for (int i = 2; i <= n; ++i) {
        fact *= static_cast<double>(i);
    }
    const double nrm = std::pow(omega / PI, 0.25) / std::sqrt(std::pow(2.0, n) * fact);
    return nrm * std::exp(-0.5 * omega * x * x) * hermite_phys(n, xi);
}

/** Isotropic HO energy of the n-th lowest state (0-based), including degeneracy. */
inline double analytic_eigen_energy(const Parameters &p, int n) {
    if (is_exact_nbody(p)) {
        if (p.lambda_contact != 0.0 || p.ee || p.trap != TrapKind::Harmonic || p.omega <= 0.0 || n < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const SpinKind spin = two_electron_spin(p);
        if (p.n_electrons == 2 && spin != SpinKind::Unspecified) {
            if (n != 0) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double e_singlet = p.omega * static_cast<double>(p.spatial_dim);
            return (spin == SpinKind::Triplet) ? (e_singlet + p.omega) : e_singlet;
        }
        if (p.fermion && p.spatial_dim == 1) {
            double E = 0.0;
            for (int k = 0; k < p.n_electrons; ++k) {
                E += p.omega * (static_cast<double>(k) + 0.5);
            }
            return E;
        }
        if (!p.fermion) {
            // Sequential deflated Lanczos with an x-shifted trial tracks the
            // Cartesian tower |n,0,…,0⟩, E = ω(n + D/2), D = n_e × dim.
            const int D = p.n_electrons * p.spatial_dim;
            return p.omega * (static_cast<double>(n) + 0.5 * static_cast<double>(D));
        }
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (p.lambda_contact != 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (p.trap != TrapKind::Harmonic || p.omega <= 0.0 || n < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const int shell = ho_cartesian_shell(n, p.spatial_dim);
    return p.omega * (static_cast<double>(shell) + 0.5 * static_cast<double>(p.spatial_dim));
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

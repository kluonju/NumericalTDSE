#pragma once

/**
 * Uniform-grid two-electron Schrödinger solver.
 *
 * Two electrons in SpatialDim = 1 or 2 are stored as one real wave function
 * on a (2*SpatialDim)-dimensional configuration-space grid:
 *
 *   SpatialDim = 1  →  ψ(x1, x2)           on N×N          (TGK08 helium model)
 *   SpatialDim = 2  →  ψ(x1, y1, x2, y2)   on N⁴           (1 particle in 4D)
 *
 * That 4D picture is the interacting Hamiltonian, not an extra approximation:
 * T = −½∇²_{r1} − ½∇²_{r2} is the 4D Laplacian, v(r1)+v(r2)+W(|r1−r2|) is a
 * local multiplicative potential, and the singlet is ψ(r1,r2)=ψ(r2,r1).
 *
 * Ground states use Strang imag-time splitting: pointwise exp(−τ V) and an
 * implicit 1D heat sweep along each configuration axis (Thomas / Dirichlet).
 */

#include "tdse/parameters.hpp"

#include "MRCPP/Printer"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tdse {

inline int invert_default_grid(const Parameters &p) {
    if (p.n_grid > 0) {
        return p.n_grid;
    }
    return (p.spatial_dim == 1) ? 49 : 15;
}

/** Force an odd count so r = 0 sits on a node. */
inline int invert_odd_grid(int n) {
    if (n < 9) {
        n = 9;
    }
    if (n % 2 == 0) {
        ++n;
    }
    return n;
}

template <int SpatialDim>
struct TwoElectronGrid {
    static_assert(SpatialDim == 1 || SpatialDim == 2, "SpatialDim must be 1 or 2");
    static constexpr int kConfigDim = 2 * SpatialDim;

    int n = 0;
    std::size_t ns = 0; ///< spatial samples, n^SpatialDim
    std::size_t nc = 0; ///< configuration samples, n^{2 SpatialDim}
    double L = 0.0;
    double dx = 0.0;
    double dV_s = 1.0; ///< spatial volume element
    double dV_c = 1.0; ///< configuration-space volume element
    double a2 = 1.0;
    bool ee = true;

    std::vector<double> axis; ///< x_i = −L + i dx
    std::vector<double> psi;
    std::vector<double> Vcfg;
    std::vector<double> work;
    std::vector<double> v_ext;
    std::vector<double> dens;

    void setup(const Parameters &p) {
        n = invert_odd_grid(invert_default_grid(p));
        L = p.L;
        if (L <= 0.0) {
            throw std::invalid_argument("invert requires MRA L > 0");
        }
        dx = 2.0 * L / static_cast<double>(n - 1);
        dV_s = (SpatialDim == 1) ? dx : (dx * dx);
        dV_c = dV_s * dV_s;
        a2 = p.soft_a * p.soft_a;
        ee = p.ee;

        ns = 1;
        nc = 1;
        for (int k = 0; k < SpatialDim; ++k) {
            ns *= static_cast<std::size_t>(n);
        }
        for (int k = 0; k < kConfigDim; ++k) {
            nc *= static_cast<std::size_t>(n);
        }

        axis.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            axis[static_cast<std::size_t>(i)] = -L + static_cast<double>(i) * dx;
        }
        psi.assign(nc, 0.0);
        Vcfg.assign(nc, 0.0);
        work.assign(nc, 0.0);
        v_ext.assign(ns, 0.0);
        dens.assign(ns, 0.0);
    }

    double coord(int i) const { return axis[static_cast<std::size_t>(i)]; }

    bool boundary_spatial(int i, int j) const {
        if (i <= 0 || i >= n - 1) {
            return true;
        }
        if constexpr (SpatialDim == 2) {
            return (j <= 0 || j >= n - 1);
        }
        return false;
    }

    std::size_t spat(int i, int j = 0) const {
        if constexpr (SpatialDim == 1) {
            (void)j;
            return static_cast<std::size_t>(i);
        }
        return static_cast<std::size_t>(i + n * j);
    }

    std::size_t cfg(int i1, int j1, int i2, int j2) const {
        if constexpr (SpatialDim == 1) {
            (void)j1;
            (void)j2;
            return static_cast<std::size_t>(i1 + n * i2);
        }
        return static_cast<std::size_t>(i1 + n * (j1 + n * (i2 + n * j2)));
    }

    std::size_t exchange_index(std::size_t p) const {
        if constexpr (SpatialDim == 1) {
            const int i1 = static_cast<int>(p % static_cast<std::size_t>(n));
            const int i2 = static_cast<int>(p / static_cast<std::size_t>(n));
            return static_cast<std::size_t>(i2 + n * i1);
        }
        const int n1 = n;
        const int i1 = static_cast<int>(p % static_cast<std::size_t>(n1));
        p /= static_cast<std::size_t>(n1);
        const int j1 = static_cast<int>(p % static_cast<std::size_t>(n1));
        p /= static_cast<std::size_t>(n1);
        const int i2 = static_cast<int>(p % static_cast<std::size_t>(n1));
        const int j2 = static_cast<int>(p / static_cast<std::size_t>(n1));
        return cfg(i2, j2, i1, j1);
    }

    double radius2(int i, int j = 0) const {
        const double x = coord(i);
        if constexpr (SpatialDim == 1) {
            (void)j;
            return x * x;
        }
        const double y = coord(j);
        return x * x + y * y;
    }

    double soft_w(double r2) const { return 1.0 / std::sqrt(r2 + a2); }

    double trap_value(const Parameters &p, double r2) const {
        switch (p.trap) {
            case TrapKind::None:
                return 0.0;
            case TrapKind::Harmonic:
                return 0.5 * p.omega * p.omega * r2;
            case TrapKind::SoftAtom:
                return -p.Z / std::sqrt(r2 + a2);
        }
        return 0.0;
    }

    void fill_trap(std::vector<double> &v, const Parameters &p) const {
        v.resize(ns);
        if constexpr (SpatialDim == 1) {
            for (int i = 0; i < n; ++i) {
                v[spat(i)] = trap_value(p, radius2(i));
            }
        } else {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    v[spat(i, j)] = trap_value(p, radius2(i, j));
                }
            }
        }
    }

    void fill_harmonic(std::vector<double> &v, double omega) const {
        v.resize(ns);
        const double k = 0.5 * omega * omega;
        if constexpr (SpatialDim == 1) {
            for (int i = 0; i < n; ++i) {
                v[spat(i)] = k * radius2(i);
            }
        } else {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    v[spat(i, j)] = k * radius2(i, j);
                }
            }
        }
    }

    void fill_atom(std::vector<double> &v, double Z) const {
        v.resize(ns);
        if constexpr (SpatialDim == 1) {
            for (int i = 0; i < n; ++i) {
                v[spat(i)] = -Z / std::sqrt(radius2(i) + a2);
            }
        } else {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    v[spat(i, j)] = -Z / std::sqrt(radius2(i, j) + a2);
                }
            }
        }
    }

    void build_config_potential() {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(nc); ++p) {
            Vcfg[static_cast<std::size_t>(p)] = config_potential(static_cast<std::size_t>(p));
        }
    }

    double config_potential(std::size_t p) const {
        if constexpr (SpatialDim == 1) {
            const int i1 = static_cast<int>(p % static_cast<std::size_t>(n));
            const int i2 = static_cast<int>(p / static_cast<std::size_t>(n));
            double w = 0.0;
            if (ee) {
                const double dx12 = coord(i1) - coord(i2);
                w = soft_w(dx12 * dx12);
            }
            return v_ext[spat(i1)] + v_ext[spat(i2)] + w;
        }
        const int n1 = n;
        int q = static_cast<int>(p);
        const int i1 = q % n1;
        q /= n1;
        const int j1 = q % n1;
        q /= n1;
        const int i2 = q % n1;
        const int j2 = q / n1;
        double w = 0.0;
        if (ee) {
            const double dx12 = coord(i1) - coord(i2);
            const double dy12 = coord(j1) - coord(j2);
            w = soft_w(dx12 * dx12 + dy12 * dy12);
        }
        return v_ext[spat(i1, j1)] + v_ext[spat(i2, j2)] + w;
    }

    void gaussian_trial(double alpha) {
        const double nrm1 = std::pow(alpha / 3.14159265358979323846, 0.25 * SpatialDim);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(nc); ++p) {
            psi[static_cast<std::size_t>(p)] = trial_value(static_cast<std::size_t>(p), alpha, nrm1);
        }
        zero_boundary();
        normalize();
        symmetrize();
        normalize();
    }

    double trial_value(std::size_t p, double alpha, double nrm1) const {
        if constexpr (SpatialDim == 1) {
            const int i1 = static_cast<int>(p % static_cast<std::size_t>(n));
            const int i2 = static_cast<int>(p / static_cast<std::size_t>(n));
            const double g1 = nrm1 * std::exp(-0.5 * alpha * radius2(i1));
            const double g2 = nrm1 * std::exp(-0.5 * alpha * radius2(i2));
            return g1 * g2;
        }
        const int n1 = n;
        int q = static_cast<int>(p);
        const int i1 = q % n1;
        q /= n1;
        const int j1 = q % n1;
        q /= n1;
        const int i2 = q % n1;
        const int j2 = q / n1;
        const double g1 = nrm1 * std::exp(-0.5 * alpha * radius2(i1, j1));
        const double g2 = nrm1 * std::exp(-0.5 * alpha * radius2(i2, j2));
        return g1 * g2;
    }

    void zero_boundary() {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(nc); ++p) {
            if (on_boundary(static_cast<std::size_t>(p))) {
                psi[static_cast<std::size_t>(p)] = 0.0;
            }
        }
    }

    bool on_boundary(std::size_t p) const {
        if constexpr (SpatialDim == 1) {
            const int i1 = static_cast<int>(p % static_cast<std::size_t>(n));
            const int i2 = static_cast<int>(p / static_cast<std::size_t>(n));
            return (i1 <= 0 || i1 >= n - 1 || i2 <= 0 || i2 >= n - 1);
        }
        const int n1 = n;
        int q = static_cast<int>(p);
        const int i1 = q % n1;
        q /= n1;
        const int j1 = q % n1;
        q /= n1;
        const int i2 = q % n1;
        const int j2 = q / n1;
        return (i1 <= 0 || i1 >= n - 1 || j1 <= 0 || j1 >= n - 1 || i2 <= 0 || i2 >= n - 1 ||
                j2 <= 0 || j2 >= n - 1);
    }

    double square_norm() const {
        double s = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : s) schedule(static)
#endif
        for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(nc); ++p) {
            const double z = psi[static_cast<std::size_t>(p)];
            s += z * z;
        }
        return s * dV_c;
    }

    void normalize() {
        const double nrm = std::sqrt(square_norm());
        if (nrm <= 0.0) {
            throw std::runtime_error("two-electron grid: cannot normalize a zero wave function");
        }
        const double inv = 1.0 / nrm;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(nc); ++p) {
            psi[static_cast<std::size_t>(p)] *= inv;
        }
    }

    void symmetrize() {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(nc); ++p) {
            const std::size_t q = exchange_index(static_cast<std::size_t>(p));
            if (static_cast<std::size_t>(p) < q) {
                const double a = 0.5 * (psi[static_cast<std::size_t>(p)] + psi[q]);
                psi[static_cast<std::size_t>(p)] = a;
                psi[q] = a;
            }
        }
    }

    void scale_potential(double tau) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(nc); ++p) {
            psi[static_cast<std::size_t>(p)] *= std::exp(-tau * Vcfg[static_cast<std::size_t>(p)]);
        }
    }

    void heat_axis(int axis, double sigma) {
        const int nn = n;
        if constexpr (SpatialDim == 1) {
            const int stride = ipow_n(axis);
            if (axis == 0) {
#ifdef _OPENMP
#pragma omp parallel
#endif
                {
                    std::vector<double> cp(static_cast<std::size_t>(nn), 0.0);
                    std::vector<double> d(static_cast<std::size_t>(nn), 0.0);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                    for (int i2 = 1; i2 < nn - 1; ++i2) {
                        thomas_line(&psi[static_cast<std::size_t>(n * i2)], 1, sigma, cp, d);
                    }
                }
            } else {
#ifdef _OPENMP
#pragma omp parallel
#endif
                {
                    std::vector<double> cp(static_cast<std::size_t>(nn), 0.0);
                    std::vector<double> d(static_cast<std::size_t>(nn), 0.0);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                    for (int i1 = 1; i1 < nn - 1; ++i1) {
                        thomas_line(&psi[static_cast<std::size_t>(i1)], stride, sigma, cp, d);
                    }
                }
            }
            return;
        } else {
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::vector<double> cp(static_cast<std::size_t>(nn), 0.0);
            std::vector<double> d(static_cast<std::size_t>(nn), 0.0);
            const int n2 = nn * nn;
            const int n3 = n2 * nn;
            if (axis == 0) {
#ifdef _OPENMP
#pragma omp for collapse(3) schedule(static)
#endif
                for (int j2 = 1; j2 < nn - 1; ++j2) {
                    for (int i2 = 1; i2 < nn - 1; ++i2) {
                        for (int j1 = 1; j1 < nn - 1; ++j1) {
                            thomas_line(&psi[static_cast<std::size_t>(n * j1 + n2 * i2 + n3 * j2)],
                                        1,
                                        sigma,
                                        cp,
                                        d);
                        }
                    }
                }
            } else if (axis == 1) {
#ifdef _OPENMP
#pragma omp for collapse(3) schedule(static)
#endif
                for (int j2 = 1; j2 < nn - 1; ++j2) {
                    for (int i2 = 1; i2 < nn - 1; ++i2) {
                        for (int i1 = 1; i1 < nn - 1; ++i1) {
                            thomas_line(&psi[static_cast<std::size_t>(i1 + n2 * i2 + n3 * j2)],
                                        nn,
                                        sigma,
                                        cp,
                                        d);
                        }
                    }
                }
            } else if (axis == 2) {
#ifdef _OPENMP
#pragma omp for collapse(3) schedule(static)
#endif
                for (int j2 = 1; j2 < nn - 1; ++j2) {
                    for (int j1 = 1; j1 < nn - 1; ++j1) {
                        for (int i1 = 1; i1 < nn - 1; ++i1) {
                            thomas_line(&psi[static_cast<std::size_t>(i1 + nn * j1 + n3 * j2)],
                                        n2,
                                        sigma,
                                        cp,
                                        d);
                        }
                    }
                }
            } else {
#ifdef _OPENMP
#pragma omp for collapse(3) schedule(static)
#endif
                for (int i2 = 1; i2 < nn - 1; ++i2) {
                    for (int j1 = 1; j1 < nn - 1; ++j1) {
                        for (int i1 = 1; i1 < nn - 1; ++i1) {
                            thomas_line(&psi[static_cast<std::size_t>(i1 + nn * j1 + n2 * i2)],
                                        n3,
                                        sigma,
                                        cp,
                                        d);
                        }
                    }
                }
            }
        }
        }
    }

    int ipow_n(int k) const {
        int s = 1;
        for (int i = 0; i < k; ++i) {
            s *= n;
        }
        return s;
    }

    void thomas_line(double *line, int stride, double sigma, std::vector<double> &cprime, std::vector<double> &d) {
        const int nn = n;
        const double a = -sigma;
        const double b = 1.0 + 2.0 * sigma;
        const double c = -sigma;
        cprime[1] = c / b;
        d[1] = line[stride] / b;
        for (int i = 2; i <= nn - 2; ++i) {
            const double denom = b - a * cprime[static_cast<std::size_t>(i - 1)];
            cprime[static_cast<std::size_t>(i)] = c / denom;
            d[static_cast<std::size_t>(i)] = (line[i * stride] - a * d[static_cast<std::size_t>(i - 1)]) / denom;
        }
        line[(nn - 2) * stride] = d[static_cast<std::size_t>(nn - 2)];
        for (int i = nn - 3; i >= 1; --i) {
            line[i * stride] = d[static_cast<std::size_t>(i)] - cprime[static_cast<std::size_t>(i)] * line[(i + 1) * stride];
        }
        line[0] = 0.0;
        line[(nn - 1) * stride] = 0.0;
    }

    void imag_step(double tau) {
        const double sigma = tau / (2.0 * dx * dx);
        scale_potential(0.5 * tau);
        for (int ax = 0; ax < kConfigDim; ++ax) {
            heat_axis(ax, sigma);
        }
        scale_potential(0.5 * tau);
        zero_boundary();
        symmetrize();
        normalize();
    }

    void apply_H(const std::vector<double> &in, std::vector<double> &out) const {
        const double ckin = -0.5 / (dx * dx);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(nc); ++p) {
            const std::size_t u = static_cast<std::size_t>(p);
            if (on_boundary(u)) {
                out[u] = 0.0;
                continue;
            }
            double lap = 0.0;
            if constexpr (SpatialDim == 1) {
                const int i1 = static_cast<int>(u % static_cast<std::size_t>(n));
                const int i2 = static_cast<int>(u / static_cast<std::size_t>(n));
                const std::size_t s0 = 1;
                const std::size_t s1 = static_cast<std::size_t>(n);
                (void)i1;
                (void)i2;
                lap += in[u - s0] - 2.0 * in[u] + in[u + s0];
                lap += in[u - s1] - 2.0 * in[u] + in[u + s1];
            } else {
                const std::size_t s0 = 1;
                const std::size_t s1 = static_cast<std::size_t>(n);
                const std::size_t s2 = s1 * s1;
                const std::size_t s3 = s2 * s1;
                lap += in[u - s0] - 2.0 * in[u] + in[u + s0];
                lap += in[u - s1] - 2.0 * in[u] + in[u + s1];
                lap += in[u - s2] - 2.0 * in[u] + in[u + s2];
                lap += in[u - s3] - 2.0 * in[u] + in[u + s3];
            }
            out[u] = ckin * lap + Vcfg[u] * in[u];
        }
    }

    double energy() {
        apply_H(psi, work);
        double e = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : e) schedule(static)
#endif
        for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(nc); ++p) {
            e += psi[static_cast<std::size_t>(p)] * work[static_cast<std::size_t>(p)];
        }
        return e * dV_c;
    }

    double residual(double E) {
        apply_H(psi, work);
        double num = 0.0;
        double den = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : num, den) schedule(static)
#endif
        for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(nc); ++p) {
            const std::size_t u = static_cast<std::size_t>(p);
            const double r = work[u] - E * psi[u];
            num += r * r;
            den += psi[u] * psi[u];
        }
        if (den <= 0.0) {
            return 0.0;
        }
        return std::sqrt(num / den);
    }

    double relax(int steps, double tau, double ethresh, int print_every = 0, const char *tag = "") {
        double E = energy();
        for (int s = 0; s < steps; ++s) {
            imag_step(tau);
            const bool check = ((s + 1) % 5 == 0 || s + 1 == steps);
            const bool report = (print_every > 0 && ((s + 1) % print_every == 0 || s + 1 == steps));
            if (check || report) {
                const double En = energy();
                if (report) {
                    std::stringstream ss;
                    if (tag != nullptr && tag[0] != '\0') {
                        ss << "  " << tag << " ";
                    }
                    ss << "step=" << (s + 1) << "/" << steps << "  E=" << std::fixed << std::setprecision(10) << En
                       << "  ||(H-E)ψ||=" << std::scientific << residual(En);
                    if (check && s >= 4) {
                        ss << "  |ΔE|=" << std::scientific << std::abs(En - E);
                    }
                    println(0, ss.str());
                }
                if (check) {
                    if (std::abs(En - E) < ethresh && s >= 4) {
                        return En;
                    }
                    E = En;
                }
            }
        }
        return energy();
    }

    void compute_density() {
        std::fill(dens.begin(), dens.end(), 0.0);
        if constexpr (SpatialDim == 1) {
            for (int i2 = 0; i2 < n; ++i2) {
                const double *row = &psi[static_cast<std::size_t>(n * i2)];
                for (int i1 = 0; i1 < n; ++i1) {
                    const double z = row[i1];
                    dens[static_cast<std::size_t>(i1)] += z * z;
                }
            }
            const double fac = 2.0 * dx;
            for (std::size_t i = 0; i < ns; ++i) {
                dens[i] *= fac;
            }
        } else {
            const int n2 = n * n;
            const int n3 = n2 * n;
            for (int j2 = 0; j2 < n; ++j2) {
                for (int i2 = 0; i2 < n; ++i2) {
                    const std::size_t blk = static_cast<std::size_t>(n2 * i2 + n3 * j2);
                    for (int j1 = 0; j1 < n; ++j1) {
                        const double *row = &psi[blk + static_cast<std::size_t>(n * j1)];
                        double *out = &dens[static_cast<std::size_t>(n * j1)];
                        for (int i1 = 0; i1 < n; ++i1) {
                            const double z = row[i1];
                            out[i1] += z * z;
                        }
                    }
                }
            }
            const double fac = 2.0 * dx * dx;
            for (std::size_t i = 0; i < ns; ++i) {
                dens[i] *= fac;
            }
        }
    }

    double density_integral() const {
        double s = 0.0;
        for (double v : dens) {
            s += v;
        }
        return s * dV_s;
    }

    double l1_density(const std::vector<double> &target) const {
        double s = 0.0;
        for (std::size_t i = 0; i < ns; ++i) {
            s += std::abs(dens[i] - target[i]);
        }
        return s * dV_s;
    }

    void hartree(std::vector<double> &vh) const {
        vh.assign(ns, 0.0);
        if constexpr (SpatialDim == 1) {
            for (int i = 0; i < n; ++i) {
                double s = 0.0;
                const double xi = coord(i);
                for (int j = 0; j < n; ++j) {
                    const double dxij = xi - coord(j);
                    s += dens[static_cast<std::size_t>(j)] * soft_w(dxij * dxij);
                }
                vh[static_cast<std::size_t>(i)] = s * dx;
            }
        } else {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    double s = 0.0;
                    const double xi = coord(i);
                    const double yi = coord(j);
                    for (int l = 0; l < n; ++l) {
                        const double yl = coord(l);
                        for (int k = 0; k < n; ++k) {
                            const double dxik = xi - coord(k);
                            const double dyil = yi - yl;
                            s += dens[spat(k, l)] * soft_w(dxik * dxik + dyil * dyil);
                        }
                    }
                    vh[spat(i, j)] = s * dx * dx;
                }
            }
        }
    }

    /** v_s = ε + (½ ∇²φ)/φ with φ = sqrt(n/2); ε absorbed later as a shift. */
    void ks_potential(std::vector<double> &vs, double ncut) const {
        vs.assign(ns, 0.0);
        std::vector<double> phi(ns, 0.0);
        for (std::size_t i = 0; i < ns; ++i) {
            const double nval = std::max(dens[i], 0.0);
            phi[i] = std::sqrt(0.5 * nval);
        }
        const double invdx2 = 1.0 / (dx * dx);
        if constexpr (SpatialDim == 1) {
            for (int i = 1; i < n - 1; ++i) {
                if (dens[static_cast<std::size_t>(i)] < ncut) {
                    continue;
                }
                const double lap = (phi[static_cast<std::size_t>(i - 1)] - 2.0 * phi[static_cast<std::size_t>(i)] +
                                    phi[static_cast<std::size_t>(i + 1)]) *
                                   invdx2;
                vs[static_cast<std::size_t>(i)] = 0.5 * lap / phi[static_cast<std::size_t>(i)];
            }
        } else {
            for (int j = 1; j < n - 1; ++j) {
                for (int i = 1; i < n - 1; ++i) {
                    const std::size_t u = spat(i, j);
                    if (dens[u] < ncut) {
                        continue;
                    }
                    const double lap = (phi[spat(i - 1, j)] + phi[spat(i + 1, j)] + phi[spat(i, j - 1)] +
                                        phi[spat(i, j + 1)] - 4.0 * phi[u]) *
                                       invdx2;
                    vs[u] = 0.5 * lap / phi[u];
                }
            }
        }
    }

    int density_peak() const {
        int imax = 0;
        double best = -1.0;
        for (std::size_t i = 0; i < ns; ++i) {
            if (dens[i] > best) {
                best = dens[i];
                imax = static_cast<int>(i);
            }
        }
        return imax;
    }
};

} // namespace tdse

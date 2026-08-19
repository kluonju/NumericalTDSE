#pragma once

/**
 * Stationary solver: ground state and a few lowest eigenstates.
 *
 * Linear H (exact N-body, or orbitals with λ = 0)
 *   Lanczos — default; ground state from a Krylov space of H, keeping the
 *             Ritz vector that overlaps the smooth trial (the MW kinetic
 *             operator is not bounded below). A few heat-kernel imag-time
 *             steps then strip high-frequency junk from ⟨H⟩. Excited states
 *             use a nodal/Hermite guess plus the same heat polish in the
 *             orthogonal complement.
 *   ITP     — Strang imag-time: exp(−Vτ/2) exp(−Tτ) exp(−Vτ/2) with the
 *             MRCPP heat kernel exp((τ/2)∇²) (smoothing). Gram–Schmidt
 *             for excited states. Krylov/RK4 of the MW Hamiltonian is
 *             not used: that operator is unbounded below.
 *
 * Nonlinear orbitals (λ ρ ≠ 0): SCF with an inner Lanczos, or orbital ITP.
 */

#include "tdse/analytic.hpp"
#include "tdse/observables.hpp"
#include "tdse/operators.hpp"
#include "tdse/parameters.hpp"
#include "tdse/parallel.hpp"
#include "tdse/propagator.hpp"
#include "tdse/setup.hpp"
#include "tdse/wavefunction.hpp"

#include "MRCPP/Printer"
#include "MRCPP/Timer"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tdse {

template <int D>
struct EigenPair {
    double energy = 0.0;
    double residual = 0.0;
    double overlap = 0.0;
    double dipole = 0.0;
    double nrm = 1.0;
    int nodes_re = 0;
    int nodes_im = 0;
    std::unique_ptr<CplxFun<D>> psi;
};

inline double residual_threshold(const Parameters &p) {
    return (p.eigen_residual > 0.0) ? p.eigen_residual : (50.0 * p.prec);
}

inline double eigen_trial_center(const Parameters &p) {
    if (p.n_states > 1 && std::abs(p.x0) < 1.0e-14 && std::abs(p.k0) < 1.0e-14) {
        return 0.7;
    }
    return p.x0;
}

template <int D>
double hamiltonian_residual(double prec,
                            CplxFun<D> &psi,
                            OperatorSet<D> &ops,
                            mrcpp::FunctionTree<D> &V,
                            double energy) {
    CplxFun<D> Hpsi(psi.mra);
    apply_hamiltonian(prec, Hpsi, ops, psi, V);
    CplxFun<D> err(psi.mra);
    add_cplx(prec, err, 1.0, Hpsi, -energy, psi);
    const double n = norm(psi);
    if (n <= 0.0) {
        return 0.0;
    }
    return norm(err) / n;
}

template <int D>
double overlap_ho_eigen(double prec, CplxFun<D> &psi, const Parameters &p, int n) {
    if (n < 0) {
        return 0.0;
    }
    if constexpr (D == 2 || D == 3) {
        if (is_hydrogenic_1e(p) && n == 0) {
            CplxFun<D> ana(psi.mra);
            auto re_f = [&](const mrcpp::Coord<D> &r) -> double { return hydrogen_1s_eval<D>(r, p.Z); };
            auto im_f = [&](const mrcpp::Coord<D> &r) -> double {
                (void)r;
                return 0.0;
            };
            mrcpp::project<D, double>(prec, ana.re, re_f);
            mrcpp::project<D, double>(prec, ana.im, im_f);
            const double nrm = norm(psi) * norm(ana);
            if (nrm <= 0.0) {
                return 0.0;
            }
            return std::abs(inner(psi, ana)) / nrm;
        }
    }
    if (p.trap != TrapKind::Harmonic || p.omega <= 0.0) {
        return 0.0;
    }
    if (p.lambda_contact != 0.0) {
        return 0.0;
    }
    // Exact N-body GS is the isotropic D-dimensional Gaussian (product of 1D HO
    // ground orbitals). Excited N-body states and interacting / fermionic
    // cases have no elementary overlap target here.
    if (p.representation == Representation::Exact && (p.ee || p.fermion)) {
        return 0.0;
    }
    CplxFun<D> ana(psi.mra);
    auto re_f = [&](const mrcpp::Coord<D> &r) -> double {
        double v = ho_eigen_1d(r[0], n, p.omega).real();
        for (int d = 1; d < D; ++d) {
            v *= ho_eigen_1d(r[d], 0, p.omega).real();
        }
        return v;
    };
    auto im_f = [&](const mrcpp::Coord<D> &r) -> double {
        (void)r;
        return 0.0;
    };
    mrcpp::project<D, double>(prec, ana.re, re_f);
    mrcpp::project<D, double>(prec, ana.im, im_f);
    const double nrm = norm(psi) * norm(ana);
    if (nrm <= 0.0) {
        return 0.0;
    }
    return std::abs(inner(psi, ana)) / nrm;
}

template <int D>
void fill_pair_observables(double prec,
                           EigenPair<D> &ep,
                           OperatorSet<D> &ops,
                           mrcpp::FunctionTree<D> &V,
                           const Parameters &p,
                           int n) {
    ep.energy = energy_expectation(prec, *ep.psi, ops, V);
    ep.residual = hamiltonian_residual(prec, *ep.psi, ops, V, ep.energy);
    ep.dipole = total_dipole<D>(prec, *ep.psi, p);
    ep.nrm = norm(*ep.psi);
    ep.nodes_re = ep.psi->re.getNNodes();
    ep.nodes_im = ep.psi->im.getNNodes();
    ep.overlap = overlap_ho_eigen(prec, *ep.psi, p, n);
}

template <int D>
void reconstruct_cplx(double prec,
                      CplxFun<D> &out,
                      const std::vector<std::unique_ptr<CplxFun<D>>> &v,
                      const Eigen::VectorXcd &coeff) {
    mrcpp::FunctionTreeVector<D> re_terms;
    mrcpp::FunctionTreeVector<D> im_terms;
    const int m = static_cast<int>(coeff.size());
    for (int k = 0; k < m; ++k) {
        const auto ck = coeff(k);
        auto *vk = v[static_cast<std::size_t>(k)].get();
        re_terms.push_back(std::make_tuple(ck.real(), &vk->re));
        im_terms.push_back(std::make_tuple(ck.real(), &vk->im));
        if (std::abs(ck.imag()) > 1.0e-18) {
            re_terms.push_back(std::make_tuple(-ck.imag(), &vk->im));
            im_terms.push_back(std::make_tuple(ck.imag(), &vk->re));
        }
    }
    mrcpp::FunctionTree<D> re_new(out.mra);
    mrcpp::FunctionTree<D> im_new(out.mra);
    mrcpp::add(prec, re_new, re_terms);
    mrcpp::add(prec, im_new, im_terms);
    copy_into(out.re, re_new);
    copy_into(out.im, im_new);
}

/**
 * Lowest eigenpair of a Hermitian MW operator from one Krylov space.
 * The MW kinetic operator is not bounded below, so the algebraically
 * smallest Ritz value is discarded; the pair kept is the one that
 * overlaps the smooth trial (`|Q(0,k)|`).
 * `trial` is orthogonalised against `deflate` and normalized in place.
 */
template <int D>
std::vector<EigenPair<D>> lanczos_lowest(double prec,
                                         CplxFun<D> &trial,
                                         const ApplyOp<D> &applyH,
                                         OperatorSet<D> &ops,
                                         mrcpp::FunctionTree<D> &V,
                                         const Parameters &p,
                                         int n_states,
                                         const std::vector<CplxFun<D> *> &deflate = {}) {
    const int m = std::max(p.krylov_dim, n_states + 2);
    if (m < 2) {
        throw std::invalid_argument("Krylov dimension must be >= 2");
    }
    orthogonalize(prec, trial, deflate);
    normalize(trial);

    {
        CplxFun<D> Ht(trial.mra);
        applyH(Ht, trial);
        println(0, "  trial Rayleigh ⟨H⟩ = " << inner(trial, Ht).real());
    }

    std::vector<std::unique_ptr<CplxFun<D>>> v;
    v.reserve(static_cast<std::size_t>(m + 1));
    v.push_back(std::make_unique<CplxFun<D>>(trial.mra));
    copy_into(*v[0], trial);

    std::vector<double> alpha(static_cast<std::size_t>(m), 0.0);
    std::vector<double> beta(static_cast<std::size_t>(m), 0.0);
    int m_eff = m;
    const double breakdown = 1.0e-14;

    for (int j = 0; j < m; ++j) {
        auto w = std::make_unique<CplxFun<D>>(trial.mra);
        applyH(*w, *v[static_cast<std::size_t>(j)]);
        orthogonalize(prec, *w, deflate);

        if (j > 0) {
            auto tmp = std::make_unique<CplxFun<D>>(trial.mra);
            add_cplx(prec, *tmp, 1.0, *w, -beta[static_cast<std::size_t>(j - 1)], *v[static_cast<std::size_t>(j - 1)]);
            w = std::move(tmp);
        }

        alpha[static_cast<std::size_t>(j)] = inner(*v[static_cast<std::size_t>(j)], *w).real();
        auto w2 = std::make_unique<CplxFun<D>>(trial.mra);
        add_cplx(prec, *w2, 1.0, *w, -alpha[static_cast<std::size_t>(j)], *v[static_cast<std::size_t>(j)]);
        w = std::move(w2);

        const double bj = norm(*w);
        beta[static_cast<std::size_t>(j)] = bj;
        if (bj < breakdown || j == m - 1) {
            m_eff = j + 1;
            break;
        }
        w->re.rescale(1.0 / bj);
        w->im.rescale(1.0 / bj);
        v.push_back(std::move(w));
    }

    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m_eff, m_eff);
    for (int i = 0; i < m_eff; ++i) {
        T(i, i) = alpha[static_cast<std::size_t>(i)];
        if (i + 1 < m_eff) {
            T(i, i + 1) = beta[static_cast<std::size_t>(i)];
            T(i + 1, i) = beta[static_cast<std::size_t>(i)];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    const Eigen::VectorXd Teval = es.eigenvalues();
    const Eigen::MatrixXd Q = es.eigenvectors();

    const double e_floor = energy_floor(p);

    std::vector<int> order(static_cast<std::size_t>(m_eff));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return std::abs(Q(0, a)) > std::abs(Q(0, b));
    });

    {
        std::stringstream ss;
        ss << "  Lanczos |Q(0,k)| vs T-Ritz:";
        println(0, ss.str());
        const int nshow = std::min(m_eff, 6);
        for (int i = 0; i < nshow; ++i) {
            const int k = order[static_cast<std::size_t>(i)];
            std::stringstream line;
            line << std::scientific << std::setprecision(6) << "    k=" << k << "  |Q0|=" << std::abs(Q(0, k))
                 << "  T-Ritz=" << Teval(k);
            println(0, line.str());
        }
    }

    EigenPair<D> best;
    bool have = false;
    double picked_q0 = 0.0;
    double picked_teval = 0.0;
    for (int k : order) {
        if (Teval(k) < e_floor) {
            continue;
        }
        best.psi = std::make_unique<CplxFun<D>>(trial.mra);
        copy_into(*best.psi, trial);
        Eigen::VectorXcd coeff = Q.col(k).cast<std::complex<double>>();
        reconstruct_cplx(prec, *best.psi, v, coeff);
        crop(*best.psi, prec);
        if (norm(*best.psi) <= 0.0) {
            continue;
        }
        normalize(*best.psi);
        orthogonalize(prec, *best.psi, deflate);
        if (norm(*best.psi) <= 0.0) {
            continue;
        }
        normalize(*best.psi);
        fill_pair_observables(prec, best, ops, V, p, 0);
        if (best.energy < e_floor) {
            continue;
        }
        picked_q0 = std::abs(Q(0, k));
        picked_teval = Teval(k);
        have = true;
        break;
    }
    if (!have) {
        throw std::runtime_error("Lanczos failed to reconstruct a physical Ritz vector");
    }
    println(0,
            "  selected |Q0|=" << picked_q0 << "  T-Ritz=" << picked_teval << "  ⟨H⟩=" << best.energy);

    std::vector<EigenPair<D>> out;
    out.push_back(std::move(best));
    (void)n_states;
    return out;
}

/** 2D isotropic HO orbital φ_{nx,ny}(x,y). */
inline double ho_orb_2d(double x, double y, int nx, int ny, double omega) {
    return ho_eigen_1d(x, nx, omega).real() * ho_eigen_1d(y, ny, omega).real();
}

template <int D>
void project_eigen_guess(double prec, CplxFun<D> &psi, const Parameters &p, int k) {
    // Two electrons in 2D: configuration (x1,y1,x2,y2). Use exchange-adapted
    // HO products so Lanczos stays in the singlet (even) or triplet (odd) sector.
    if constexpr (D == 4) {
        const SpinKind spin = two_electron_spin(p);
        if (p.n_electrons == 2 && p.spatial_dim == 2 && p.omega > 0.0 && k >= 0 &&
            spin != SpinKind::Unspecified &&
            (p.trap == TrapKind::Harmonic || p.trap == TrapKind::SoftAtom)) {
            const double w = p.omega;
            const double s2 = 1.0 / std::sqrt(2.0);
            auto re_f = [&](const mrcpp::Coord<4> &r) -> double {
                const double x1 = r[0], y1 = r[1], x2 = r[2], y2 = r[3];
                const auto g = [&](int nx, int ny, double x, double y) {
                    return ho_orb_2d(x, y, nx, ny, w);
                };
                if (spin == SpinKind::Singlet) {
                    if (k == 0) {
                        return g(0, 0, x1, y1) * g(0, 0, x2, y2);
                    }
                    // First even excitation: symmetrized |px, s>
                    return s2 * (g(1, 0, x1, y1) * g(0, 0, x2, y2) + g(0, 0, x1, y1) * g(1, 0, x2, y2));
                }
                // Triplet: antisymmetrized |px, s> (lowest odd), then |py, s>
                if (k == 0) {
                    return s2 * (g(1, 0, x1, y1) * g(0, 0, x2, y2) - g(0, 0, x1, y1) * g(1, 0, x2, y2));
                }
                return s2 * (g(0, 1, x1, y1) * g(0, 0, x2, y2) - g(0, 0, x1, y1) * g(0, 1, x2, y2));
            };
            auto im_f = [&](const mrcpp::Coord<4> &r) -> double {
                (void)r;
                return 0.0;
            };
            mrcpp::project<4, double>(prec, psi.re, re_f);
            mrcpp::project<4, double>(prec, psi.im, im_f);
            return;
        }
    }
    if constexpr (D == 2 || D == 3) {
        if (p.trap == TrapKind::SoftAtom && p.n_electrons == 1) {
            auto re_f = [&](const mrcpp::Coord<D> &r) -> double {
                if (k <= 0) {
                    return hydrogen_1s_eval<D>(r, p.Z);
                }
                double r2 = 0.0;
                for (int d = 0; d < D; ++d) {
                    r2 += r[d] * r[d];
                }
                double poly = 1.0;
                for (int i = 0; i < k; ++i) {
                    poly *= r[0];
                }
                const double zeta = (D == 2) ? (2.0 * p.Z) : p.Z;
                return poly * std::exp(-zeta * std::sqrt(r2));
            };
            auto im_f = [&](const mrcpp::Coord<D> &r) -> double {
                (void)r;
                return 0.0;
            };
            mrcpp::project<D, double>(prec, psi.re, re_f);
            mrcpp::project<D, double>(prec, psi.im, im_f);
            return;
        }
    }
    if (p.trap == TrapKind::Harmonic && p.omega > 0.0 && k >= 0) {
        auto re_f = [&](const mrcpp::Coord<D> &r) -> double {
            double v = ho_eigen_1d(r[0], k, p.omega).real();
            for (int d = 1; d < D; ++d) {
                v *= ho_eigen_1d(r[d], 0, p.omega).real();
            }
            return v;
        };
        auto im_f = [&](const mrcpp::Coord<D> &r) -> double {
            (void)r;
            return 0.0;
        };
        mrcpp::project<D, double>(prec, psi.re, re_f);
        mrcpp::project<D, double>(prec, psi.im, im_f);
        return;
    }
    if (k <= 0) {
        project_psi(prec, psi, p, eigen_trial_center(p));
        return;
    }
    if constexpr (D == 1) {
        auto re_f = [&](const mrcpp::Coord<1> &r) -> double {
            const double nrm = std::pow(p.alpha / PI, 0.25);
            double pk = 1.0;
            for (int i = 0; i < k; ++i) {
                pk *= r[0];
            }
            return nrm * std::exp(-0.5 * p.alpha * r[0] * r[0]) * pk;
        };
        auto im_f = [&](const mrcpp::Coord<1> &r) -> double {
            (void)r;
            return 0.0;
        };
        mrcpp::project<1, double>(prec, psi.re, re_f);
        mrcpp::project<1, double>(prec, psi.im, im_f);
    } else {
        project_psi(prec, psi, p, eigen_trial_center(p) + 0.55 * static_cast<double>(k));
    }
}

template <int D>
void heat_polish(double prec,
                 CplxFun<D> &psi,
                 OperatorSet<D> &ops,
                 mrcpp::FunctionTree<D> &V,
                 mrcpp::HeatOperator<D> &heat,
                 double dt,
                 int nsteps,
                 const std::vector<CplxFun<D> *> &deflate) {
    for (int s = 0; s < nsteps; ++s) {
        orthogonalize(prec, psi, deflate);
        step_split_imag(prec, psi, ops, V, dt, &heat);
        crop(psi, prec);
        if (norm(psi) < 1.0e-18) {
            throw std::runtime_error("heat polish collapsed the wave function; decrease polish τ");
        }
        normalize(psi);
        orthogonalize(prec, psi, deflate);
        normalize(psi);
    }
}

template <int D>
std::vector<EigenPair<D>> lanczos_spectrum(double prec,
                                           OperatorSet<D> &ops,
                                           mrcpp::FunctionTree<D> &V,
                                           const Parameters &p,
                                           int n_states) {
    ApplyOp<D> H = [&](CplxFun<D> &out, CplxFun<D> &in) { apply_hamiltonian(prec, out, ops, in, V); };
    std::vector<EigenPair<D>> pairs;
    std::vector<CplxFun<D> *> defs;
    const double tau = std::min(0.10, std::max(p.dt, 0.02));
    const int polish = std::min(8, std::max(2, p.eigen_maxiter));
    // 4D heat-kernel convolutions are too expensive for a smoke/ground run;
    // Lanczos without polish is still well-defined.
    std::unique_ptr<mrcpp::HeatOperator<D>> heat;
    if constexpr (D <= 3) {
        heat = std::make_unique<mrcpp::HeatOperator<D>>(ops.mra, 0.5 * tau, prec);
    }
    for (int k = 0; k < n_states; ++k) {
        CplxFun<D> trial(ops.mra);
        project_eigen_guess(prec, trial, p, k);
        orthogonalize(prec, trial, defs);
        normalize(trial);
        auto one = lanczos_lowest(prec, trial, H, ops, V, p, 1, defs);
        if (one.empty()) {
            throw std::runtime_error("Lanczos returned no state");
        }
        if constexpr (D <= 3) {
            // Heat polish smears the Coulomb cusp; hydrogenic 1s is already the trial.
            if (!is_hydrogenic_1e(p)) {
                heat_polish(prec, *one[0].psi, ops, V, *heat, tau, polish, defs);
            }
        }
        fill_pair_observables(prec, one[0], ops, V, p, k);
        pairs.push_back(std::move(one[0]));
        defs.push_back(pairs.back().psi.get());
    }
    return pairs;
}

template <int D>
std::vector<CplxFun<D> *> as_ptrs(std::vector<std::unique_ptr<CplxFun<D>>> &states) {
    std::vector<CplxFun<D> *> ptrs;
    ptrs.reserve(states.size());
    for (auto &s : states) {
        ptrs.push_back(s.get());
    }
    return ptrs;
}

template <int D>
std::vector<EigenPair<D>> itp_lowest(double prec,
                                     OperatorSet<D> &ops,
                                     mrcpp::FunctionTree<D> &V,
                                     const Parameters &p,
                                     int n_states,
                                     std::ostream *history) {
    const int nsteps = std::max(1, std::min(p.eigen_maxiter, static_cast<int>(std::llround(p.T / p.dt))));
    if (p.dt <= 0.0) {
        throw std::invalid_argument("ITP requires dt > 0");
    }
    if constexpr (D > 3) {
        throw std::invalid_argument(
                "ITP HeatOperator is not used for MRA dimension 4; set method = 'lanczos'");
    }
    mrcpp::print::header(0, "Building HeatOperator exp((Δτ/2) ∇²) = exp(−T Δτ)");
    mrcpp::HeatOperator<D> heat(ops.mra, 0.5 * p.dt, prec);
    std::vector<std::unique_ptr<CplxFun<D>>> found;
    std::vector<EigenPair<D>> out;

    for (int k = 0; k < n_states; ++k) {
        auto psi = std::make_unique<CplxFun<D>>(ops.mra);
        project_eigen_guess(prec, *psi, p, k);
        normalize(*psi);
        auto prev = as_ptrs(found);
        orthogonalize(prec, *psi, prev);
        normalize(*psi);

        double E_prev = 0.0;
        int used = 0;
        CplxFun<D> best(ops.mra);
        copy_into(best, *psi);
        double E_best = energy_expectation(prec, *psi, ops, V);
        const double floor = energy_floor(p);
        for (int s = 0; s < nsteps; ++s) {
            orthogonalize(prec, *psi, prev);
            step_split_imag(prec, *psi, ops, V, p.dt, &heat);
            crop(*psi, prec);
            if (norm(*psi) < 1.0e-18) {
                throw std::runtime_error("imaginary-time step collapsed the wave function; decrease dt");
            }
            normalize(*psi);
            orthogonalize(prec, *psi, prev);
            normalize(*psi);

            const double E = energy_expectation(prec, *psi, ops, V);
            const double res = hamiltonian_residual(prec, *psi, ops, V, E);
            ++used;

            if (std::isfinite(E) && E >= floor && E < E_best) {
                E_best = E;
                copy_into(best, *psi);
            }
            const bool diverged = !std::isfinite(E) || E < floor || res > 50.0;
            if (diverged) {
                println(0, "  ITP n=" << k << " diverged at step " << (s + 1) << "; restoring lowest energy");
                copy_into(*psi, best);
                normalize(*psi);
                break;
            }

            if (history != nullptr && parallel::io_rank() && k == 0) {
                Observables o;
                o.t = (s + 1) * p.dt;
                o.nrm = norm(*psi);
                o.dipole = total_dipole<D>(prec, *psi, p);
                o.energy = E;
                o.residual = res;
                o.energy_analytic = analytic_eigen_energy(p, k);
                o.overlap_analytic = overlap_ho_eigen(prec, *psi, p, k);
                o.n_nodes_re = psi->re.getNNodes();
                o.n_nodes_im = psi->im.getNNodes();
                write_row(*history, o);
                print_row(o);
            } else if (s % std::max(1, p.print_every) == 0) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(8) << "  ITP n=" << k << "  step=" << (s + 1)
                   << "  E=" << E << "  ||(H-E)ψ||=" << std::scientific << res;
                println(0, ss.str());
            }

            // MW residual is not a reliable stop; |ΔE| and overlap are.
            const double ov = (history != nullptr) ? overlap_ho_eigen(prec, *psi, p, k) : 0.0;
            if (s > 0 && std::abs(E - E_prev) < p.eigen_thr) {
                break;
            }
            if (ov > 0.995) {
                break;
            }
            E_prev = E;
        }
        println(0, "  ITP state " << k << " used " << used << " imaginary-time steps");

        EigenPair<D> ep;
        ep.psi = std::move(psi);
        fill_pair_observables(prec, ep, ops, V, p, k);
        found.push_back(std::make_unique<CplxFun<D>>(ops.mra));
        copy_into(*found.back(), *ep.psi);
        out.push_back(std::move(ep));
    }
    return out;
}

inline void print_spectrum(const Parameters &p, const std::vector<Observables> &rows) {
    mrcpp::print::header(0, "Eigenvalues");
    println(0, "  n            E_num            E_ana       residual     |⟨num|ana⟩|");
    for (const auto &o : rows) {
        std::stringstream ss;
        ss << "  " << std::setw(3) << o.state << "  " << std::scientific << std::setprecision(10) << o.energy
           << "  ";
        if (std::isfinite(o.energy_analytic)) {
            ss << o.energy_analytic;
        } else {
            ss << "            —";
        }
        ss << "  " << std::setprecision(3) << o.residual;
        if (o.overlap_analytic > 0.0) {
            ss << "  " << std::setprecision(10) << o.overlap_analytic;
        }
        println(0, ss.str());
    }
    mrcpp::print::separator(0, '=', 2);
    (void)p;
}

template <int D>
std::vector<Observables> pairs_to_rows(const std::vector<EigenPair<D>> &pairs, const Parameters &p) {
    std::vector<Observables> rows;
    rows.reserve(pairs.size());
    for (int k = 0; k < static_cast<int>(pairs.size()); ++k) {
        const auto &ep = pairs[static_cast<std::size_t>(k)];
        Observables o;
        o.state = k;
        o.t = static_cast<double>(k);
        o.energy = ep.energy;
        o.residual = ep.residual;
        o.overlap_analytic = ep.overlap;
        o.energy_analytic = analytic_eigen_energy(p, k);
        o.dipole = ep.dipole;
        o.nrm = ep.nrm;
        o.n_nodes_re = ep.nodes_re;
        o.n_nodes_im = ep.nodes_im;
        rows.push_back(o);
    }
    return rows;
}

template <int D>
void write_spectrum_file(const Parameters &p, const std::vector<Observables> &rows) {
    if (!parallel::io_rank()) {
        return;
    }
    std::ofstream csv(p.output);
    if (!csv) {
        std::cerr << "NumericalTDSE error: cannot open output file: " << p.output << std::endl;
        parallel::abort_all(1);
    }
    write_spectrum_header(csv);
    for (const auto &o : rows) {
        write_spectrum_row(csv, o);
    }
}

template <int D>
void plot_eigenstates(const std::vector<EigenPair<D>> &pairs, const Parameters &p) {
    for (int k = 0; k < static_cast<int>(pairs.size()); ++k) {
        maybe_plot(*pairs[static_cast<std::size_t>(k)].psi, p, "n" + std::to_string(k));
    }
}

template <int D>
int simulate_stationary_exact(const Parameters &p) {
    mrcpp::Timer timer;
    if (parallel::size > 1) {
        println(0,
                "  note: exact stationary uses one FunctionTree; extra MPI ranks "
                "replicate the work. Use OpenMP for this mode.");
    }
    auto MRA = make_mra<D>(p);
    MRA.print();

    OperatorSet<D> ops(MRA, p);
    TimeDependentPotential<D> pot(p);
    auto V = project_V(MRA, p.prec, pot, 0.0);
    println(0, "  V nodes         : " << V->getNNodes());
    if (p.trap == TrapKind::SoftAtom) {
        println(0, "  nuclear trap    : −Z/sqrt(r²+a²)  Z=" << p.Z << "  a=" << p.soft_a
                                                             << "  (adaptive MW refines at the nucleus)");
    }

    const double center = eigen_trial_center(p);
    if (std::abs(center - p.x0) > 1.0e-12) {
        println(0, "  note: even trial (x0 = 0) misses odd eigenstates; using x0 = " << center);
    }

    CplxFun<D> trial(MRA);
    project_psi(p.prec, trial, p, center);
    normalize(trial);
    identity_sanity_check(p.prec, ops, trial);

    std::vector<EigenPair<D>> pairs;
    const bool write_itp_history = (p.eigen_method == EigenMethod::Itp && p.n_states == 1);

    std::ofstream hist;
    if (write_itp_history && parallel::io_rank()) {
        hist.open(p.output);
        if (!hist) {
            std::cerr << "NumericalTDSE error: cannot open output file: " << p.output << std::endl;
            parallel::abort_all(1);
        }
        write_header(hist);
    }
    parallel::barrier();

    mrcpp::print::header(0, (p.job == JobKind::Ground) ? "Ground state" : "Lowest eigenstates");
    println(0, "  method          : " << eigen_method_name(p.eigen_method));
    mrcpp::print::value(0, "n_states", static_cast<double>(p.n_states));

    if (p.eigen_method == EigenMethod::Lanczos) {
        pairs = lanczos_spectrum(p.prec, ops, *V, p, p.n_states);
    } else {
        pairs = itp_lowest<D>(p.prec, ops, *V, p, p.n_states, write_itp_history ? &hist : nullptr);
    }
    if (hist.is_open()) {
        hist.close();
    }

    auto rows = pairs_to_rows(pairs, p);
    print_spectrum(p, rows);
    if (!write_itp_history) {
        write_spectrum_file<D>(p, rows);
    }
    plot_eigenstates(pairs, p);

    const double res_thr = residual_threshold(p);
    for (const auto &o : rows) {
        if (o.overlap_analytic > 0.99) {
            continue;
        }
        if (std::isfinite(o.energy_analytic) && std::abs(o.energy - o.energy_analytic) < 0.05) {
            continue;
        }
        if (o.residual > std::max(res_thr, 1.0)) {
            println(0,
                    "  warning: state " << o.state << " residual " << o.residual
                                        << " > threshold " << res_thr
                                        << " (increase krylov_dim, or tighten prec)");
        }
    }
    mrcpp::print::footer(0, timer, 2);
    return 0;
}

template <int D>
std::unique_ptr<mrcpp::FunctionTree<D>>
orbital_potential(const mrcpp::MultiResolutionAnalysis<D> &MRA,
                  const Parameters &p,
                  mrcpp::FunctionTree<D> &Vext,
                  const std::vector<std::unique_ptr<CplxFun<D>>> &orbs) {
    auto V = std::make_unique<mrcpp::FunctionTree<D>>(MRA);
    if (p.lambda_contact == 0.0) {
        Vext.deep_copy(V.get());
        return V;
    }
    mrcpp::FunctionTree<D> rho(MRA);
    rho.setZero();
    bool first = true;
    for (const auto &phi : orbs) {
        if (!phi) {
            continue;
        }
        mrcpp::FunctionTree<D> re2(MRA);
        mrcpp::FunctionTree<D> im2(MRA);
        mrcpp::square(p.prec, re2, phi->re);
        mrcpp::square(p.prec, im2, phi->im);
        mrcpp::FunctionTree<D> dens_i(MRA);
        mrcpp::add(p.prec, dens_i, 1.0, re2, 1.0, im2);
        if (first) {
            dens_i.deep_copy(&rho);
            first = false;
        } else {
            rho.add(1.0, dens_i);
        }
    }
    mrcpp::FunctionTree<D> VH(MRA);
    rho.deep_copy(&VH);
    VH.rescale(p.lambda_contact);
    mrcpp::add(p.prec, *V, 1.0, Vext, 1.0, VH);
    return V;
}

template <int D>
int simulate_stationary_orbitals(const Parameters &p) {
    mrcpp::Timer timer;
    if (parallel::size > 1) {
        println(0,
                "  note: orbital eigenstates keep every orbital on every rank "
                "(Gram–Schmidt). Use OpenMP; extra MPI ranks replicate the work.");
    }
    auto MRA = make_mra<D>(p);
    MRA.print();
    OperatorSet<D> ops(MRA, p);
    TimeDependentPotential<D> pot(p);
    auto Vext = project_V(MRA, p.prec, pot, 0.0);

    const int n = p.n_states;
    std::vector<std::unique_ptr<CplxFun<D>>> orbs(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        orbs[static_cast<std::size_t>(i)] = std::make_unique<CplxFun<D>>(MRA);
        double c = eigen_trial_center(p);
        if (n > 1) {
            c = eigen_trial_center(p) * (i == 0 ? 1.0 : (2.0 * i / (n - 1) - 1.0));
            if (std::abs(c) < 1.0e-12) {
                c = 0.55 * (i + 1);
            }
        }
        project_psi(p.prec, *orbs[static_cast<std::size_t>(i)], p, c);
        normalize(*orbs[static_cast<std::size_t>(i)]);
    }

    mrcpp::print::header(0, "Orbital ground / eigenstates");
    println(0, "  method          : " << eigen_method_name(p.eigen_method));
    mrcpp::print::value(0, "n_states", static_cast<double>(n));

    std::vector<EigenPair<D>> pairs;
    const int nscf = (p.lambda_contact == 0.0 && p.eigen_method == EigenMethod::Lanczos) ? 1 : p.eigen_maxiter;
    double E_prev = 0.0;
    std::unique_ptr<mrcpp::HeatOperator<D>> heat;
    if (p.eigen_method == EigenMethod::Itp) {
        if (p.dt <= 0.0) {
            throw std::invalid_argument("ITP requires dt > 0");
        }
        mrcpp::print::header(0, "Building HeatOperator exp((Δτ/2) ∇²) = exp(−T Δτ)");
        heat = std::make_unique<mrcpp::HeatOperator<D>>(MRA, 0.5 * p.dt, p.prec);
    }

    for (int it = 0; it < nscf; ++it) {
        auto V = orbital_potential(MRA, p, *Vext, orbs);
        if (p.eigen_method == EigenMethod::Lanczos) {
            pairs = lanczos_spectrum(p.prec, ops, *V, p, n);
            for (int i = 0; i < n; ++i) {
                copy_into(*orbs[static_cast<std::size_t>(i)], *pairs[static_cast<std::size_t>(i)].psi);
            }
        } else {
            for (int i = 0; i < n; ++i) {
                std::vector<CplxFun<D> *> lower;
                for (int j = 0; j < i; ++j) {
                    lower.push_back(orbs[static_cast<std::size_t>(j)].get());
                }
                orthogonalize(p.prec, *orbs[static_cast<std::size_t>(i)], lower);
                step_split_imag(p.prec, *orbs[static_cast<std::size_t>(i)], ops, *V, p.dt, heat.get());
                crop(*orbs[static_cast<std::size_t>(i)], p.prec);
                normalize(*orbs[static_cast<std::size_t>(i)]);
                orthogonalize(p.prec, *orbs[static_cast<std::size_t>(i)], lower);
                normalize(*orbs[static_cast<std::size_t>(i)]);
            }
            pairs.clear();
            for (int i = 0; i < n; ++i) {
                EigenPair<D> ep;
                ep.psi = std::make_unique<CplxFun<D>>(MRA);
                copy_into(*ep.psi, *orbs[static_cast<std::size_t>(i)]);
                fill_pair_observables(p.prec, ep, ops, *V, p, i);
                pairs.push_back(std::move(ep));
            }
        }

        double Etot = 0.0;
        for (const auto &ep : pairs) {
            Etot += ep.energy;
        }
        println(0, "  SCF/ITP iter " << (it + 1) << "  Σ ε = " << std::setprecision(10) << Etot);
        if (it > 0 && std::abs(Etot - E_prev) < p.eigen_thr) {
            break;
        }
        E_prev = Etot;
        if (p.lambda_contact == 0.0 && p.eigen_method == EigenMethod::Lanczos) {
            break;
        }
    }

    auto rows = pairs_to_rows(pairs, p);
    if (p.lambda_contact != 0.0) {
        for (auto &o : rows) {
            o.energy_analytic = std::numeric_limits<double>::quiet_NaN();
            o.overlap_analytic = 0.0;
        }
    }
    print_spectrum(p, rows);
    write_spectrum_file<D>(p, rows);
    plot_eigenstates(pairs, p);
    mrcpp::print::footer(0, timer, 2);
    return 0;
}

inline int run_stationary(const Parameters &p) {
    if (p.representation == Representation::Orbital) {
        switch (p.spatial_dim) {
            case 1:
                return simulate_stationary_orbitals<1>(p);
            case 2:
                return simulate_stationary_orbitals<2>(p);
            case 3:
                return simulate_stationary_orbitals<3>(p);
            default:
                throw std::invalid_argument("spatial_dim must be 1, 2 or 3");
        }
    }
    const int D = mra_dimension(p);
    switch (D) {
        case 1:
            return simulate_stationary_exact<1>(p);
        case 2:
            return simulate_stationary_exact<2>(p);
        case 3:
            return simulate_stationary_exact<3>(p);
        case 4:
            return simulate_stationary_exact<4>(p);
        default:
            throw std::invalid_argument("exact N-body MRA dimension must be 1, 2, 3 or 4");
    }
}

} // namespace tdse

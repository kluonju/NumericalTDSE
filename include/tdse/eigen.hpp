#pragma once

/**
 * Stationary solver: ground state and a few lowest eigenstates.
 *
 * Linear H (exact N-body, or orbitals with λ = 0)
 *   Lanczos — default; ground state from a Krylov space of H, keeping the
 *             Ritz vector that overlaps the smooth trial (the MW kinetic
 *             operator is not bounded below). Excited states use a
 *             nodal/Hermite guess plus a few imaginary-time RK4 steps in
 *             the orthogonal complement.
 *   ITP     — imaginary-time filter ∂τψ = −Hψ with Gram–Schmidt. If
 *             propagator='krylov', RK4 imag is used instead of exp(−Hτ).
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
    if (p.trap != TrapKind::Harmonic || p.omega <= 0.0 || n < 0) {
        return 0.0;
    }
    if (p.representation == Representation::Exact && p.n_electrons != 1) {
        return 0.0;
    }
    if (p.lambda_contact != 0.0) {
        return 0.0;
    }
    CplxFun<D> ana(psi.mra);
    if constexpr (D == 1) {
        auto re_f = [&](const mrcpp::Coord<1> &r) -> double { return ho_eigen_1d(r[0], n, p.omega).real(); };
        auto im_f = [&](const mrcpp::Coord<1> &r) -> double { return 0.0; };
        mrcpp::project<1, double>(prec, ana.re, re_f);
        mrcpp::project<1, double>(prec, ana.im, im_f);
    } else {
        if (n != 0) {
            return 0.0;
        }
        auto re_f = [&](const mrcpp::Coord<D> &r) -> double {
            double r2 = 0.0;
            for (int d = 0; d < D; ++d) {
                r2 += r[d] * r[d];
            }
            const double nrm = std::pow(p.omega / PI, 0.25 * static_cast<double>(D));
            return nrm * std::exp(-0.5 * p.omega * r2);
        };
        auto im_f = [&](const mrcpp::Coord<D> &r) -> double {
            (void)r;
            return 0.0;
        };
        mrcpp::project<D, double>(prec, ana.re, re_f);
        mrcpp::project<D, double>(prec, ana.im, im_f);
    }
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

    double e_floor = -1.0;
    if (p.trap == TrapKind::Harmonic) {
        e_floor = -0.05;
    } else if (p.trap == TrapKind::SoftAtom) {
        e_floor = -0.5 * p.Z * p.Z - 10.0;
    }

    struct Cand {
        int k = 0;
        double q0 = 0.0;
        EigenPair<D> ep;
    };
    std::vector<Cand> cands;
    cands.reserve(static_cast<std::size_t>(m_eff));
    for (int k = 0; k < m_eff; ++k) {
        Cand c;
        c.k = k;
        c.q0 = std::abs(Q(0, k));
        c.ep.psi = std::make_unique<CplxFun<D>>(trial.mra);
        copy_into(*c.ep.psi, trial);
        Eigen::VectorXcd coeff = Q.col(k).cast<std::complex<double>>();
        reconstruct_cplx(prec, *c.ep.psi, v, coeff);
        crop(*c.ep.psi, prec);
        if (norm(*c.ep.psi) <= 0.0) {
            continue;
        }
        normalize(*c.ep.psi);
        orthogonalize(prec, *c.ep.psi, deflate);
        if (norm(*c.ep.psi) <= 0.0) {
            continue;
        }
        normalize(*c.ep.psi);
        fill_pair_observables(prec, c.ep, ops, V, p, 0);
        cands.push_back(std::move(c));
    }

    std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) { return a.q0 > b.q0; });
    EigenPair<D> best;
    bool have = false;
    for (auto &c : cands) {
        if (c.ep.energy < e_floor) {
            continue;
        }
        best = std::move(c.ep);
        have = true;
        break;
    }
    if (!have && !cands.empty()) {
        best = std::move(cands.front().ep);
        have = true;
    }
    if (!have) {
        throw std::runtime_error("Lanczos failed to reconstruct a Ritz vector");
    }

    std::vector<EigenPair<D>> out;
    out.push_back(std::move(best));
    (void)n_states;
    (void)Teval;
    return out;
}

template <int D>
void project_eigen_guess(double prec, CplxFun<D> &psi, const Parameters &p, int k) {
    if (k <= 0) {
        project_psi(prec, psi, p, eigen_trial_center(p));
        return;
    }
    if constexpr (D == 1) {
        auto re_f = [&](const mrcpp::Coord<1> &r) -> double {
            if (p.trap == TrapKind::Harmonic && p.omega > 0.0) {
                return ho_eigen_1d(r[0], k, p.omega).real();
            }
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
std::vector<EigenPair<D>> lanczos_spectrum(double prec,
                                           OperatorSet<D> &ops,
                                           mrcpp::FunctionTree<D> &V,
                                           const Parameters &p,
                                           int n_states) {
    ApplyOp<D> H = [&](CplxFun<D> &out, CplxFun<D> &in) { apply_hamiltonian(prec, out, ops, in, V); };
    std::vector<EigenPair<D>> pairs;
    std::vector<CplxFun<D> *> defs;
    const int polish = std::min(4, p.eigen_maxiter);
    for (int k = 0; k < n_states; ++k) {
        CplxFun<D> trial(ops.mra);
        project_eigen_guess(prec, trial, p, k);
        orthogonalize(prec, trial, defs);
        normalize(trial);
        if (k == 0) {
            auto one = lanczos_lowest(prec, trial, H, ops, V, p, 1, defs);
            if (one.empty()) {
                throw std::runtime_error("Lanczos returned no state");
            }
            one[0].overlap = overlap_ho_eigen(prec, *one[0].psi, p, k);
            pairs.push_back(std::move(one[0]));
        } else {
            // Higher states: imag-time polish in the orthogonal complement.
            // A large Krylov space of the MW kinetic operator is not bounded below.
            for (int s = 0; s < polish; ++s) {
                orthogonalize(prec, trial, defs);
                step_rk4_imag(prec, trial, ops, V, p.dt);
                crop(trial, prec);
                normalize(trial);
            }
            orthogonalize(prec, trial, defs);
            normalize(trial);
            EigenPair<D> ep;
            ep.psi = std::make_unique<CplxFun<D>>(ops.mra);
            copy_into(*ep.psi, trial);
            fill_pair_observables(prec, ep, ops, V, p, k);
            pairs.push_back(std::move(ep));
        }
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
    const double res_thr = residual_threshold(p);
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
        for (int s = 0; s < nsteps; ++s) {
            orthogonalize(prec, *psi, prev);
            if (p.propagator == Propagator::Krylov) {
                // exp(−Hτ) on the MW kinetic operator amplifies spurious negative
                // modes; RK4 on ∂τψ = −Hψ stays in the basin of the trial.
                step_rk4_imag(prec, *psi, ops, V, p.dt);
            } else {
                step_imaginary(prec, *psi, ops, V, p);
            }
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

            if (s > 0 && std::abs(E - E_prev) < p.eigen_thr && res < res_thr) {
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

    for (int it = 0; it < nscf; ++it) {
        auto V = orbital_potential(MRA, p, *Vext, orbs);
        if (p.eigen_method == EigenMethod::Lanczos) {
            pairs = lanczos_spectrum(p.prec, ops, *V, p, n);
            for (int i = 0; i < n; ++i) {
                copy_into(*orbs[static_cast<std::size_t>(i)], *pairs[static_cast<std::size_t>(i)].psi);
            }
        } else {
            auto prev_all = as_ptrs(orbs);
            for (int i = 0; i < n; ++i) {
                std::vector<CplxFun<D> *> lower;
                for (int j = 0; j < i; ++j) {
                    lower.push_back(orbs[static_cast<std::size_t>(j)].get());
                }
                orthogonalize(p.prec, *orbs[static_cast<std::size_t>(i)], lower);
                if (p.propagator == Propagator::Krylov) {
                    step_rk4_imag(p.prec, *orbs[static_cast<std::size_t>(i)], ops, *V, p.dt);
                } else {
                    step_imaginary(p.prec, *orbs[static_cast<std::size_t>(i)], ops, *V, p);
                }
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
        default:
            throw std::invalid_argument("exact N-body MRA dimension must be 1, 2 or 3");
    }
}

} // namespace tdse

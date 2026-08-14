#pragma once

/**
 * Drive one TDSE run: build the MRA, project ψ(0) and V(0), propagate, write observables.
 */

#include "tdse/analytic.hpp"
#include "tdse/observables.hpp"
#include "tdse/operators.hpp"
#include "tdse/parameters.hpp"
#include "tdse/parallel.hpp"
#include "tdse/propagator.hpp"
#include "tdse/wavefunction.hpp"

#include "MRCPP/MWFunctions"
#include "MRCPP/MWOperators"
#include "MRCPP/Plotter"
#include "MRCPP/Printer"
#include "MRCPP/Timer"

#include "operators/TimeEvolutionOperator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tdse {

template <int D>
mrcpp::MultiResolutionAnalysis<D> make_mra(const Parameters &p) {
    std::array<int, D> corner{};
    std::array<int, D> boxes{};
    std::array<double, D> sfac{};
    corner.fill(-1);
    boxes.fill(2);
    sfac.fill(p.L);
    const mrcpp::BoundingBox<D> world(0, corner, boxes, sfac);
    if (p.use_legendre) {
        const mrcpp::LegendreBasis basis(p.order);
        return mrcpp::MultiResolutionAnalysis<D>(world, basis, p.max_depth);
    }
    const mrcpp::InterpolatingBasis basis(p.order);
    return mrcpp::MultiResolutionAnalysis<D>(world, basis, p.max_depth);
}

template <int D>
void project_psi(double prec, CplxFun<D> &psi, const Parameters &p, double center_x) {
    InitialWavefunctionReal<D> re_fun(p, center_x);
    InitialWavefunctionImag<D> im_fun(p, center_x);
    mrcpp::project<D, double>(prec, psi.re, re_fun);
    mrcpp::project<D, double>(prec, psi.im, im_fun);
}

template <int D>
std::unique_ptr<mrcpp::FunctionTree<D>>
project_V(const mrcpp::MultiResolutionAnalysis<D> &mra,
          double prec,
          TimeDependentPotential<D> &pot,
          double t) {
    auto V = std::make_unique<mrcpp::FunctionTree<D>>(mra);
    pot.set_time(t);
    mrcpp::project<D, double>(prec, *V, pot);
    return V;
}

template <int D>
void maybe_plot(CplxFun<D> &psi, const Parameters &p, const std::string &tag) {
    if (p.plot_prefix.empty() || !parallel::io_rank()) {
        return;
    }
    if constexpr (D == 1) {
        mrcpp::Coord<1> origin{-p.L};
        mrcpp::Coord<1> length{2.0 * p.L};
        mrcpp::Plotter<1> plot(origin);
        plot.setRange(length);
        plot.linePlot({p.n_plot_points}, psi.re, p.plot_prefix + "_" + tag + "_re");
        plot.linePlot({p.n_plot_points}, psi.im, p.plot_prefix + "_" + tag + "_im");
        println(0, "  wrote 1D line plots with prefix '" << p.plot_prefix << "_" << tag << "'");
    }
}

template <int D>
void identity_sanity_check(double prec, OperatorSet<D> &ops, CplxFun<D> &psi) {
    if (!ops.identity) {
        return;
    }
    mrcpp::FunctionTree<D> Ipsi(ops.mra);
    mrcpp::apply(prec, Ipsi, *ops.identity, psi.re);
    mrcpp::FunctionTree<D> err(ops.mra);
    mrcpp::add(prec, err, 1.0, Ipsi, -1.0, psi.re);
    const double abs_err = std::sqrt(std::max(0.0, err.getSquareNorm()));
    mrcpp::print::value(0, "||I*Re(ψ) − Re(ψ)|| (IdentityConvolution)", abs_err);
}

template <int D>
double free_particle_overlap(double prec, CplxFun<D> &psi, const Parameters &p, double t) {
    if constexpr (D != 1) {
        return 0.0;
    } else {
        CplxFun<1> ana(psi.mra);
        auto re_f = [&](const mrcpp::Coord<1> &r) -> double {
            return free_gaussian_1d(r[0], p.x0, p.alpha, t).real();
        };
        auto im_f = [&](const mrcpp::Coord<1> &r) -> double {
            return free_gaussian_1d(r[0], p.x0, p.alpha, t).imag();
        };
        mrcpp::project<1, double>(prec, ana.re, re_f);
        mrcpp::project<1, double>(prec, ana.im, im_f);
        const double n = norm(psi) * norm(ana);
        if (n <= 0.0) {
            return 0.0;
        }
        return std::abs(inner(psi, ana)) / n;
    }
}

template <int D>
void advance(CplxFun<D> &psi,
             OperatorSet<D> &ops,
             TimeDependentPotential<D> &pot,
             double t,
             const Parameters &p,
             mrcpp::ConvolutionOperator<D> *ReE,
             mrcpp::ConvolutionOperator<D> *ImE) {
    const double prec = p.prec;
    const double dt = p.dt;
    auto V_t = project_V(psi.mra, prec, pot, t);

    switch (p.propagator) {
        case Propagator::RK4: {
            auto V_h = project_V(psi.mra, prec, pot, t + 0.5 * dt);
            auto V_f = project_V(psi.mra, prec, pot, t + dt);
            step_rk4(prec, psi, ops, *V_t, *V_h, *V_f, dt);
            break;
        }
        case Propagator::Krylov:
            step_krylov(prec, psi, ops, *V_t, dt, p.krylov_dim);
            break;
        case Propagator::Split: {
            auto V_n = project_V(psi.mra, prec, pot, t + dt);
            step_split(prec, psi, ops, *V_t, *V_n, dt, ReE, ImE);
            break;
        }
    }
    crop(psi, prec);
    if (p.renormalize) {
        normalize(psi);
    }
}

template <int D>
int simulate_exact(const Parameters &p) {
    mrcpp::Timer timer;
    if (parallel::size > 1) {
        println(0,
                "  note: exact N-body uses one FunctionTree; MPI does not "
                "domain-decompose it. Extra ranks replicate the work. "
                "Use OpenMP (nthreads / OMP_NUM_THREADS) for this mode, "
                "and MPI for mode = 'orbital'.");
    }
    auto MRA = make_mra<D>(p);
    MRA.print();

    OperatorSet<D> ops(MRA, p);
    TimeDependentPotential<D> pot(p);

    CplxFun<D> psi(MRA);
    project_psi(p.prec, psi, p, p.x0);
    normalize(psi);

    identity_sanity_check(p.prec, ops, psi);
    maybe_plot(psi, p, "t0");

    std::unique_ptr<mrcpp::TimeEvolutionOperator<1>> teo_re;
    std::unique_ptr<mrcpp::TimeEvolutionOperator<1>> teo_im;
    mrcpp::ConvolutionOperator<D> *ReE = nullptr;
    mrcpp::ConvolutionOperator<D> *ImE = nullptr;
    if constexpr (D == 1) {
        if (p.propagator == Propagator::Split && p.use_legendre) {
            mrcpp::print::header(0, "Building TimeEvolutionOperator exp(i (Δt/2) ∂_x²)");
            const double tau = 0.5 * p.dt;
            teo_re = std::make_unique<mrcpp::TimeEvolutionOperator<1>>(
                    MRA, p.prec, tau, p.teo_finest_scale, false, p.teo_jpower);
            teo_im = std::make_unique<mrcpp::TimeEvolutionOperator<1>>(
                    MRA, p.prec, tau, p.teo_finest_scale, true, p.teo_jpower);
            ReE = teo_re.get();
            ImE = teo_im.get();
            mrcpp::print::footer(0, timer, 2);
        }
    }

    std::ofstream csv;
    if (parallel::io_rank()) {
        csv.open(p.output);
        if (!csv) {
            std::cerr << "NumericalTDSE error: cannot open output file: " << p.output << std::endl;
            parallel::abort_all(1);
        }
        write_header(csv);
    }
    parallel::barrier();

    const int nsteps = static_cast<int>(std::llround(p.T / p.dt));
    mrcpp::print::header(0, "Time evolution");
    for (int s = 0; s <= nsteps; ++s) {
        const double t = s * p.dt;
        if (s % p.print_every == 0 || s == nsteps) {
            auto V = project_V(MRA, p.prec, pot, t);
            Observables o = compute_observables(p.prec, t, psi, ops, *V, p);
            if (p.validate_free) {
                o.overlap_analytic = free_particle_overlap(p.prec, psi, p, t);
            }
            if (parallel::io_rank()) {
                write_row(csv, o);
            }
            print_row(o);
        }
        if (s == nsteps) {
            break;
        }
        advance(psi, ops, pot, t, p, ReE, ImE);
    }
    if (parallel::io_rank()) {
        csv.close();
    }
    maybe_plot(psi, p, "tT");
    mrcpp::print::footer(0, timer, 2);
    return 0;
}

/** Mean-field (contact) orbital TDSE, N ≤ 4, each orbital on FunctionTree<D>. */
template <int D>
int simulate_orbitals(const Parameters &p) {
    mrcpp::Timer timer;
    auto MRA = make_mra<D>(p);
    MRA.print();

    OperatorSet<D> ops(MRA, p);
    TimeDependentPotential<D> pot(p);
    const int n = p.n_electrons;

    std::vector<std::unique_ptr<CplxFun<D>>> orbs(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (!parallel::owns_orbital(i)) {
            continue;
        }
        auto phi = std::make_unique<CplxFun<D>>(MRA);
        double c = p.x0;
        if (n > 1) {
            c = p.x0 * (2.0 * i / (n - 1) - 1.0);
        }
        project_psi(p.prec, *phi, p, c);
        normalize(*phi);
        orbs[static_cast<std::size_t>(i)] = std::move(phi);
    }

    std::ofstream csv;
    if (parallel::io_rank()) {
        csv.open(p.output);
        if (!csv) {
            std::cerr << "NumericalTDSE error: cannot open output file: " << p.output << std::endl;
            parallel::abort_all(1);
        }
        write_header(csv);
    }
    parallel::barrier();

    auto density = [&]() {
        auto rho = std::make_unique<mrcpp::FunctionTree<D>>(MRA);
        rho->setZero();
        bool first = true;
        for (auto &phi : orbs) {
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
                dens_i.deep_copy(rho.get());
                first = false;
            } else {
                rho->add(1.0, dens_i);
            }
        }
        parallel::sum_tree<D>(*rho, p.prec);
        return rho;
    };

    const int nsteps = static_cast<int>(std::llround(p.T / p.dt));
    mrcpp::print::header(0, "Orbital time evolution (contact Hartree λ ρ)");
    if (parallel::size > 1) {
        println(0,
                "  MPI: " << parallel::size << " ranks, orbitals round-robin "
                                              "(OpenMP inside each tree)");
    }
    for (int s = 0; s <= nsteps; ++s) {
        const double t = s * p.dt;
        auto Vext = project_V(MRA, p.prec, pot, t);
        auto rho = density();
        mrcpp::FunctionTree<D> V(MRA);
        if (p.lambda_contact != 0.0) {
            mrcpp::FunctionTree<D> VH(MRA);
            rho->deep_copy(&VH);
            VH.rescale(p.lambda_contact);
            mrcpp::add(p.prec, V, 1.0, *Vext, 1.0, VH);
        } else {
            Vext->deep_copy(&V);
        }

        if (s % p.print_every == 0 || s == nsteps) {
            Observables o;
            o.t = t;
            o.n_nodes_re = 0;
            o.n_nodes_im = 0;
            for (auto &phi : orbs) {
                if (!phi) {
                    continue;
                }
                o.nrm += square_norm(*phi);
                o.dipole += dipole_moment<D>(p.prec, *phi, 0);
                o.energy += energy_expectation(p.prec, *phi, ops, V);
                o.n_nodes_re += phi->re.getNNodes();
                o.n_nodes_im += phi->im.getNNodes();
            }
            parallel::sum(o.nrm);
            parallel::sum(o.dipole);
            parallel::sum(o.energy);
            parallel::sum(o.n_nodes_re);
            parallel::sum(o.n_nodes_im);
            o.nrm = std::sqrt(std::max(0.0, o.nrm));
            if (parallel::io_rank()) {
                write_row(csv, o);
            }
            print_row(o);
        }
        if (s == nsteps) {
            break;
        }

        for (auto &phi : orbs) {
            if (!phi) {
                continue;
            }
            switch (p.propagator) {
                case Propagator::RK4: {
                    auto Vext_h = project_V(MRA, p.prec, pot, t + 0.5 * p.dt);
                    auto Vext_f = project_V(MRA, p.prec, pot, t + p.dt);
                    step_rk4(p.prec, *phi, ops, V, *Vext_h, *Vext_f, p.dt);
                    break;
                }
                case Propagator::Krylov:
                    step_krylov(p.prec, *phi, ops, V, p.dt, p.krylov_dim);
                    break;
                case Propagator::Split: {
                    auto Vext_n = project_V(MRA, p.prec, pot, t + p.dt);
                    step_split(p.prec, *phi, ops, V, *Vext_n, p.dt);
                    break;
                }
            }
            crop(*phi, p.prec);
            if (p.renormalize) {
                normalize(*phi);
            }
        }
    }
    if (parallel::io_rank()) {
        csv.close();
    }
    mrcpp::print::footer(0, timer, 2);
    return 0;
}

inline int run(const Parameters &p) {
    print_parameters(p);
    if (p.representation == Representation::Orbital) {
        switch (p.spatial_dim) {
            case 1:
                return simulate_orbitals<1>(p);
            case 2:
                return simulate_orbitals<2>(p);
            case 3:
                return simulate_orbitals<3>(p);
            default:
                throw std::invalid_argument("spatial_dim must be 1, 2 or 3");
        }
    }
    const int D = mra_dimension(p);
    switch (D) {
        case 1:
            return simulate_exact<1>(p);
        case 2:
            return simulate_exact<2>(p);
        case 3:
            return simulate_exact<3>(p);
        default:
            throw std::invalid_argument("exact N-body MRA dimension must be 1, 2 or 3");
    }
}

} // namespace tdse

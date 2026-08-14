#pragma once

/**
 * Shared MRA / projection helpers used by real-time and stationary drivers.
 */

#include "tdse/analytic.hpp"
#include "tdse/operators.hpp"
#include "tdse/parameters.hpp"
#include "tdse/parallel.hpp"
#include "tdse/wavefunction.hpp"

#include "MRCPP/MWFunctions"
#include "MRCPP/MWOperators"
#include "MRCPP/Plotter"
#include "MRCPP/Printer"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>

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

} // namespace tdse

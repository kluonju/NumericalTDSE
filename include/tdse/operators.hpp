#pragma once

/**
 * MW operators that realise the Hamiltonian pieces.
 *
 * Kinetic energy T = −½ ∇² is applied as a product of first-derivative
 * operators (ABGV by default). IdentityConvolution is a narrow-Gaussian
 * approximation of the Dirac identity and is used as a sanity check.
 * TimeEvolutionOperator is the free-particle semigroup
 *     exp(i τ ∂_x²) = exp(−i T Δt)    with τ = Δt / 2
 * (1D, Legendre scaling functions) and is the convolution analogue of
 * IdentityConvolution for the Schrödinger equation.
 */

#include "tdse/parameters.hpp"
#include "tdse/wavefunction.hpp"

#include "MRCPP/MWFunctions"
#include "MRCPP/MWOperators"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tdse {

template <int D>
class OperatorSet {
public:
    const mrcpp::MultiResolutionAnalysis<D> &mra;
    const Parameters &p;

    std::unique_ptr<mrcpp::IdentityConvolution<D>> identity;
    std::unique_ptr<mrcpp::DerivativeConvolution<D>> dconv;
    std::unique_ptr<mrcpp::DerivativeOperator<D>> deriv;

    OperatorSet(const mrcpp::MultiResolutionAnalysis<D> &mra_, const Parameters &p_)
            : mra(mra_)
            , p(p_) {
        switch (p_.kinetic) {
            case KineticKind::ABGV:
                deriv = std::make_unique<mrcpp::ABGVOperator<D>>(mra_, 0.0, 0.0);
                break;
            case KineticKind::BS:
                deriv = std::make_unique<mrcpp::BSOperator<D>>(mra_, 1);
                break;
            case KineticKind::DConv:
                deriv = std::make_unique<mrcpp::ABGVOperator<D>>(mra_, 0.0, 0.0);
                dconv = std::make_unique<mrcpp::DerivativeConvolution<D>>(mra_, p_.prec);
                break;
        }
        if (p_.ident_check) {
            identity = std::make_unique<mrcpp::IdentityConvolution<D>>(mra_, p_.prec);
        }
    }
};

/** out = −½ ∇² inp. `out` must be undefined. */
template <int D>
void apply_kinetic(double prec,
                   mrcpp::FunctionTree<D> &out,
                   OperatorSet<D> &ops,
                   mrcpp::FunctionTree<D> &inp) {
    if (ops.p.kinetic == KineticKind::DConv && D == 1 && ops.dconv) {
        mrcpp::FunctionTree<D> d1(ops.mra);
        mrcpp::apply(prec, d1, *ops.dconv, inp);
        mrcpp::FunctionTree<D> d2(ops.mra);
        mrcpp::apply(prec, d2, *ops.dconv, d1);
        d2.deep_copy(&out);
        out.rescale(-0.5);
        return;
    }

    std::vector<std::unique_ptr<mrcpp::FunctionTree<D>>> second;
    mrcpp::FunctionTreeVector<D> terms;
    second.reserve(static_cast<std::size_t>(D));

    for (int dir = 0; dir < D; ++dir) {
        mrcpp::FunctionTree<D> d1(ops.mra);
        mrcpp::apply(d1, *ops.deriv, inp, dir);
        auto d2 = std::make_unique<mrcpp::FunctionTree<D>>(ops.mra);
        mrcpp::apply(*d2, *ops.deriv, d1, dir);
        terms.push_back(std::make_tuple(-0.5, d2.get()));
        second.push_back(std::move(d2));
    }
    mrcpp::add(prec, out, terms);
}

template <int D>
void apply_kinetic_cplx(double prec, CplxFun<D> &out, OperatorSet<D> &ops, CplxFun<D> &psi) {
    apply_kinetic(prec, out.re, ops, psi.re);
    apply_kinetic(prec, out.im, ops, psi.im);
}

/** Hψ = Tψ + Vψ with real multiplication potential. `out` must be undefined. */
template <int D>
void apply_hamiltonian(double prec,
                       CplxFun<D> &out,
                       OperatorSet<D> &ops,
                       CplxFun<D> &psi,
                       mrcpp::FunctionTree<D> &V) {
    mrcpp::FunctionTree<D> Tre(ops.mra);
    mrcpp::FunctionTree<D> Tim(ops.mra);
    apply_kinetic(prec, Tre, ops, psi.re);
    apply_kinetic(prec, Tim, ops, psi.im);

    mrcpp::FunctionTree<D> Vre(ops.mra);
    mrcpp::FunctionTree<D> Vim(ops.mra);
    mrcpp::multiply(prec, Vre, 1.0, V, psi.re);
    mrcpp::multiply(prec, Vim, 1.0, V, psi.im);

    mrcpp::add(prec, out.re, 1.0, Tre, 1.0, Vre);
    mrcpp::add(prec, out.im, 1.0, Tim, 1.0, Vim);
}

/** Local unitary ψ ← [cos(Vτ) − i sin(Vτ)] ψ = exp(−i V τ) ψ. */
template <int D>
void apply_potential_exponential(double prec,
                                 CplxFun<D> &psi,
                                 mrcpp::FunctionTree<D> &V,
                                 double tau) {
    mrcpp::FunctionTree<D> cosV(V.getMRA());
    mrcpp::FunctionTree<D> sinV(V.getMRA());
    mrcpp::treeMap(prec, cosV, V, [tau](double v) { return std::cos(v * tau); });
    mrcpp::treeMap(prec, sinV, V, [tau](double v) { return std::sin(v * tau); });

    mrcpp::FunctionTree<D> c_re(V.getMRA());
    mrcpp::FunctionTree<D> s_im(V.getMRA());
    mrcpp::FunctionTree<D> s_re(V.getMRA());
    mrcpp::FunctionTree<D> c_im(V.getMRA());
    mrcpp::multiply(prec, c_re, 1.0, cosV, psi.re);
    mrcpp::multiply(prec, s_im, 1.0, sinV, psi.im);
    mrcpp::multiply(prec, s_re, 1.0, sinV, psi.re);
    mrcpp::multiply(prec, c_im, 1.0, cosV, psi.im);

    mrcpp::FunctionTree<D> re_new(V.getMRA());
    mrcpp::FunctionTree<D> im_new(V.getMRA());
    mrcpp::add(prec, re_new, 1.0, c_re, 1.0, s_im);
    mrcpp::add(prec, im_new, -1.0, s_re, 1.0, c_im);

    copy_into(psi.re, re_new);
    copy_into(psi.im, im_new);
}

/**
 * Apply the complex convolution E = ReE + i ImE:
 *   (ReE + i ImE)(ψr + i ψi) = (ReE ψr − ImE ψi) + i (ImE ψr + ReE ψi)
 */
template <int D>
void apply_complex_convolution(double prec,
                               CplxFun<D> &out,
                               mrcpp::ConvolutionOperator<D> &ReE,
                               mrcpp::ConvolutionOperator<D> &ImE,
                               CplxFun<D> &inp) {
    mrcpp::FunctionTree<D> rr(inp.mra);
    mrcpp::FunctionTree<D> ii(inp.mra);
    mrcpp::FunctionTree<D> ir(inp.mra);
    mrcpp::FunctionTree<D> ri(inp.mra);
    mrcpp::apply(prec, rr, ReE, inp.re);
    mrcpp::apply(prec, ii, ImE, inp.im);
    mrcpp::apply(prec, ir, ImE, inp.re);
    mrcpp::apply(prec, ri, ReE, inp.im);
    mrcpp::add(prec, out.re, 1.0, rr, -1.0, ii);
    mrcpp::add(prec, out.im, 1.0, ir, 1.0, ri);
}

} // namespace tdse

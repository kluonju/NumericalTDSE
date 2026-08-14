#pragma once

/**
 * Time propagators for i ∂t ψ = H ψ  (atomic units, ħ = 1).
 *
 * 1. Split (Strang)
 *    ψ(t+Δt) = exp(−i V(t+Δt) Δt/2) exp(−i T Δt) exp(−i V(t) Δt/2) ψ(t)
 *    The kinetic exponential is the MW free-particle semigroup
 *    TimeEvolutionOperator (1D, Legendre) or a Krylov exponential of T.
 *
 * 2. Krylov / short iterative Lanczos (SIL)
 *    Build an m-dimensional Krylov space of H and replace exp(−i H Δt)
 *    by exp(−i T_m Δt) on that subspace. Unitarity is preserved up to
 *    the Krylov truncation.
 *
 * 3. RK4
 *    Classical four-stage Runge–Kutta on ∂t ψ = −i H(t) ψ.
 *    Explicit, not unitary; use a smaller Δt or enable --renormalize.
 */

#include "tdse/operators.hpp"
#include "tdse/parameters.hpp"
#include "tdse/wavefunction.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tdse {

enum class TimeKind {
    Real, ///< exp(−i A Δt)
    Imag  ///< exp(−A τ)  (imaginary-time / filter)
};

template <int D>
using ApplyOp = std::function<void(CplxFun<D> &, CplxFun<D> &)>;

template <int D>
void add_cplx(double prec,
              CplxFun<D> &out,
              double a,
              CplxFun<D> &x,
              double b,
              CplxFun<D> &y) {
    mrcpp::add(prec, out.re, a, x.re, b, y.re);
    mrcpp::add(prec, out.im, a, x.im, b, y.im);
}

/** f(ψ) = −i Hψ, i.e. (f_re, f_im) = (H_im, −H_re). */
template <int D>
void minus_i_times(double prec, CplxFun<D> &out, CplxFun<D> &Hpsi) {
    Hpsi.im.deep_copy(&out.re);
    Hpsi.re.deep_copy(&out.im);
    out.im.rescale(-1.0);
    (void)prec;
}

/**
 * Short iterative Lanczos: ψ ← exp(−i A Δt) ψ for a Hermitian MW operator A
 * given as applyA(out, in) with `out` undefined on entry.
 */
template <int D>
void expm_krylov(double prec,
                 CplxFun<D> &psi,
                 const ApplyOp<D> &applyA,
                 double dt,
                 int m,
                 TimeKind kind = TimeKind::Real,
                 double eval_floor = -1.0e300) {
    if (m < 2) {
        throw std::invalid_argument("Krylov dimension must be >= 2");
    }
    const double nrm0 = norm(psi);
    if (nrm0 <= 0.0) {
        return;
    }

    std::vector<std::unique_ptr<CplxFun<D>>> v;
    v.reserve(static_cast<std::size_t>(m + 1));
    v.push_back(std::make_unique<CplxFun<D>>(psi.mra));
    copy_into(*v[0], psi);
    v[0]->re.rescale(1.0 / nrm0);
    v[0]->im.rescale(1.0 / nrm0);

    std::vector<double> alpha(static_cast<std::size_t>(m), 0.0);
    std::vector<double> beta(static_cast<std::size_t>(m), 0.0);

    int m_eff = m;
    const double breakdown = 1.0e-14;

    for (int j = 0; j < m; ++j) {
        auto w = std::make_unique<CplxFun<D>>(psi.mra);
        applyA(*w, *v[static_cast<std::size_t>(j)]);

        // w ← w − β_{j−1} v_{j−1}
        if (j > 0) {
            auto tmp = std::make_unique<CplxFun<D>>(psi.mra);
            add_cplx(prec, *tmp, 1.0, *w, -beta[static_cast<std::size_t>(j - 1)], *v[static_cast<std::size_t>(j - 1)]);
            w = std::move(tmp);
        }

        alpha[static_cast<std::size_t>(j)] = inner(*v[static_cast<std::size_t>(j)], *w).real();

        auto w2 = std::make_unique<CplxFun<D>>(psi.mra);
        add_cplx(prec, *w2, 1.0, *w, -alpha[static_cast<std::size_t>(j)], *v[static_cast<std::size_t>(j)]);

        const double bj = norm(*w2);
        beta[static_cast<std::size_t>(j)] = bj;
        if (bj < breakdown || j == m - 1) {
            m_eff = j + 1;
            break;
        }
        w2->re.rescale(1.0 / bj);
        w2->im.rescale(1.0 / bj);
        v.push_back(std::move(w2));
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
    const Eigen::VectorXd eval = es.eigenvalues();
    const Eigen::MatrixXd Q = es.eigenvectors();

    Eigen::VectorXcd coeff(m_eff);
    coeff.setZero();
    for (int k = 0; k < m_eff; ++k) {
        std::complex<double> acc = 0.0;
        for (int j = 0; j < m_eff; ++j) {
            std::complex<double> phase(0.0, 0.0);
            if (kind == TimeKind::Imag) {
                // MW kinetic is not bounded below; exp(+|λ|τ) on those
                // Ritz values grows junk. Kill anything under the physical floor.
                phase = (eval(j) < eval_floor)
                                ? std::complex<double>(0.0, 0.0)
                                : std::exp(std::complex<double>(-eval(j) * dt, 0.0));
            } else {
                phase = std::exp(std::complex<double>(0.0, -eval(j) * dt));
            }
            acc += Q(k, j) * phase * Q(0, j);
        }
        coeff(k) = nrm0 * acc;
    }

    // Reconstruct ψ = Σ_k coeff_k v_k  (real and imaginary parts separately).
    // Re(c v) = Re(c) Re(v) − Im(c) Im(v),  Im(c v) = Re(c) Im(v) + Im(c) Re(v).
    mrcpp::FunctionTreeVector<D> re_terms;
    mrcpp::FunctionTreeVector<D> im_terms;
    for (int k = 0; k < m_eff; ++k) {
        const auto ck = coeff(k);
        auto *vk = v[static_cast<std::size_t>(k)].get();
        re_terms.push_back(std::make_tuple(ck.real(), &vk->re));
        im_terms.push_back(std::make_tuple(ck.real(), &vk->im));
        if (std::abs(ck.imag()) > 1.0e-18) {
            re_terms.push_back(std::make_tuple(-ck.imag(), &vk->im));
            im_terms.push_back(std::make_tuple(ck.imag(), &vk->re));
        }
    }

    mrcpp::FunctionTree<D> re_new(psi.mra);
    mrcpp::FunctionTree<D> im_new(psi.mra);
    mrcpp::add(prec, re_new, re_terms);
    mrcpp::add(prec, im_new, im_terms);
    copy_into(psi.re, re_new);
    copy_into(psi.im, im_new);
}

template <int D>
void step_krylov(double prec, CplxFun<D> &psi, OperatorSet<D> &ops, mrcpp::FunctionTree<D> &V, double dt, int m) {
    ApplyOp<D> H = [&](CplxFun<D> &out, CplxFun<D> &in) { apply_hamiltonian(prec, out, ops, in, V); };
    expm_krylov(prec, psi, H, dt, m);
}

template <int D>
void step_krylov_imag(double prec, CplxFun<D> &psi, OperatorSet<D> &ops, mrcpp::FunctionTree<D> &V, double dt, int m) {
    ApplyOp<D> H = [&](CplxFun<D> &out, CplxFun<D> &in) { apply_hamiltonian(prec, out, ops, in, V); };
    expm_krylov(prec, psi, H, dt, m, TimeKind::Imag, energy_floor(ops.p));
}

/** RK4 on ∂_τ ψ = −H ψ (static V). */
template <int D>
void step_rk4_imag(double prec, CplxFun<D> &psi, OperatorSet<D> &ops, mrcpp::FunctionTree<D> &V, double dt) {
    auto f = [&](CplxFun<D> &k, CplxFun<D> &y) {
        CplxFun<D> Hy(psi.mra);
        apply_hamiltonian(prec, Hy, ops, y, V);
        Hy.re.deep_copy(&k.re);
        Hy.im.deep_copy(&k.im);
        k.re.rescale(-1.0);
        k.im.rescale(-1.0);
    };

    auto y_plus = [&](CplxFun<D> &out, CplxFun<D> &y, CplxFun<D> &k, double s) {
        add_cplx(prec, out, 1.0, y, s, k);
    };

    CplxFun<D> k1(psi.mra);
    f(k1, psi);

    CplxFun<D> y2(psi.mra);
    y_plus(y2, psi, k1, 0.5 * dt);
    CplxFun<D> k2(psi.mra);
    f(k2, y2);

    CplxFun<D> y3(psi.mra);
    y_plus(y3, psi, k2, 0.5 * dt);
    CplxFun<D> k3(psi.mra);
    f(k3, y3);

    CplxFun<D> y4(psi.mra);
    y_plus(y4, psi, k3, dt);
    CplxFun<D> k4(psi.mra);
    f(k4, y4);

    mrcpp::FunctionTreeVector<D> re_terms;
    mrcpp::FunctionTreeVector<D> im_terms;
    re_terms.push_back(std::make_tuple(1.0, &psi.re));
    re_terms.push_back(std::make_tuple(dt / 6.0, &k1.re));
    re_terms.push_back(std::make_tuple(dt / 3.0, &k2.re));
    re_terms.push_back(std::make_tuple(dt / 3.0, &k3.re));
    re_terms.push_back(std::make_tuple(dt / 6.0, &k4.re));
    im_terms.push_back(std::make_tuple(1.0, &psi.im));
    im_terms.push_back(std::make_tuple(dt / 6.0, &k1.im));
    im_terms.push_back(std::make_tuple(dt / 3.0, &k2.im));
    im_terms.push_back(std::make_tuple(dt / 3.0, &k3.im));
    im_terms.push_back(std::make_tuple(dt / 6.0, &k4.im));

    mrcpp::FunctionTree<D> re_new(psi.mra);
    mrcpp::FunctionTree<D> im_new(psi.mra);
    mrcpp::add(prec, re_new, re_terms);
    mrcpp::add(prec, im_new, im_terms);
    copy_into(psi.re, re_new);
    copy_into(psi.im, im_new);
}

template <int D>
void step_split_imag(double prec,
                     CplxFun<D> &psi,
                     OperatorSet<D> &ops,
                     mrcpp::FunctionTree<D> &V,
                     double dt,
                     mrcpp::HeatOperator<D> *heat = nullptr) {
    apply_potential_damp(prec, psi, V, 0.5 * dt);
    if (heat != nullptr) {
        apply_heat_kinetic(prec, psi, *heat);
    } else {
        ApplyOp<D> T = [&](CplxFun<D> &out, CplxFun<D> &in) { apply_kinetic_cplx(prec, out, ops, in); };
        expm_krylov(prec, psi, T, dt, ops.p.krylov_dim, TimeKind::Imag, energy_floor(ops.p));
    }
    apply_potential_damp(prec, psi, V, 0.5 * dt);
}

/** One imaginary-time step with the namelist propagator (TEO is not used). */
template <int D>
void step_imaginary(double prec,
                    CplxFun<D> &psi,
                    OperatorSet<D> &ops,
                    mrcpp::FunctionTree<D> &V,
                    const Parameters &p) {
    switch (p.propagator) {
        case Propagator::RK4:
            step_rk4_imag(prec, psi, ops, V, p.dt);
            break;
        case Propagator::Split:
            step_split_imag(prec, psi, ops, V, p.dt);
            break;
        case Propagator::Krylov:
            step_krylov_imag(prec, psi, ops, V, p.dt, p.krylov_dim);
            break;
    }
}

template <int D>
void step_rk4(double prec,
              CplxFun<D> &psi,
              OperatorSet<D> &ops,
              mrcpp::FunctionTree<D> &V_t,
              mrcpp::FunctionTree<D> &V_half,
              mrcpp::FunctionTree<D> &V_full,
              double dt) {
    auto f = [&](CplxFun<D> &k, CplxFun<D> &y, mrcpp::FunctionTree<D> &V) {
        CplxFun<D> Hy(psi.mra);
        apply_hamiltonian(prec, Hy, ops, y, V);
        minus_i_times(prec, k, Hy);
    };

    auto y_plus = [&](CplxFun<D> &out, CplxFun<D> &y, CplxFun<D> &k, double s) {
        add_cplx(prec, out, 1.0, y, s, k);
    };

    CplxFun<D> k1(psi.mra);
    f(k1, psi, V_t);

    CplxFun<D> y2(psi.mra);
    y_plus(y2, psi, k1, 0.5 * dt);
    CplxFun<D> k2(psi.mra);
    f(k2, y2, V_half);

    CplxFun<D> y3(psi.mra);
    y_plus(y3, psi, k2, 0.5 * dt);
    CplxFun<D> k3(psi.mra);
    f(k3, y3, V_half);

    CplxFun<D> y4(psi.mra);
    y_plus(y4, psi, k3, dt);
    CplxFun<D> k4(psi.mra);
    f(k4, y4, V_full);

    // ψ ← ψ + (dt/6) (k1 + 2 k2 + 2 k3 + k4)
    mrcpp::FunctionTreeVector<D> re_terms;
    mrcpp::FunctionTreeVector<D> im_terms;
    re_terms.push_back(std::make_tuple(1.0, &psi.re));
    re_terms.push_back(std::make_tuple(dt / 6.0, &k1.re));
    re_terms.push_back(std::make_tuple(dt / 3.0, &k2.re));
    re_terms.push_back(std::make_tuple(dt / 3.0, &k3.re));
    re_terms.push_back(std::make_tuple(dt / 6.0, &k4.re));
    im_terms.push_back(std::make_tuple(1.0, &psi.im));
    im_terms.push_back(std::make_tuple(dt / 6.0, &k1.im));
    im_terms.push_back(std::make_tuple(dt / 3.0, &k2.im));
    im_terms.push_back(std::make_tuple(dt / 3.0, &k3.im));
    im_terms.push_back(std::make_tuple(dt / 6.0, &k4.im));

    mrcpp::FunctionTree<D> re_new(psi.mra);
    mrcpp::FunctionTree<D> im_new(psi.mra);
    mrcpp::add(prec, re_new, re_terms);
    mrcpp::add(prec, im_new, im_terms);
    copy_into(psi.re, re_new);
    copy_into(psi.im, im_new);
}

/** Kinetic exponential exp(−i T Δt): MW semigroup if given, otherwise Krylov on T. */
template <int D>
void apply_kinetic_exponential(double prec,
                               CplxFun<D> &psi,
                               OperatorSet<D> &ops,
                               double dt,
                               mrcpp::ConvolutionOperator<D> *ReE = nullptr,
                               mrcpp::ConvolutionOperator<D> *ImE = nullptr) {
    if (ReE != nullptr && ImE != nullptr) {
        CplxFun<D> out(psi.mra);
        apply_complex_convolution(prec, out, *ReE, *ImE, psi);
        copy_into(psi, out);
        return;
    }
    ApplyOp<D> T = [&](CplxFun<D> &out, CplxFun<D> &in) { apply_kinetic_cplx(prec, out, ops, in); };
    expm_krylov(prec, psi, T, dt, ops.p.krylov_dim);
}

template <int D>
void step_split(double prec,
                CplxFun<D> &psi,
                OperatorSet<D> &ops,
                mrcpp::FunctionTree<D> &V_t,
                mrcpp::FunctionTree<D> &V_next,
                double dt,
                mrcpp::ConvolutionOperator<D> *ReE = nullptr,
                mrcpp::ConvolutionOperator<D> *ImE = nullptr) {
    apply_potential_exponential(prec, psi, V_t, 0.5 * dt);
    apply_kinetic_exponential(prec, psi, ops, dt, ReE, ImE);
    apply_potential_exponential(prec, psi, V_next, 0.5 * dt);
}

} // namespace tdse

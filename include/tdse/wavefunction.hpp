#pragma once

/**
 * Complex wave function as a pair of real FunctionTrees (Re ψ, Im ψ).
 *
 * This is the representation used by MRCPP's complex convolution apply
 * (ComplexObject<FunctionTree<D>>) and matches the official
 * schrodinger_semigroup1d example. FunctionTree is neither copyable nor
 * movable, so this wrapper is also non-copyable; use copy_into() instead.
 */

#include "MRCPP/MWFunctions"

#include <cmath>
#include <complex>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tdse {

template <int D>
struct CplxFun {
    const mrcpp::MultiResolutionAnalysis<D> &mra;
    mrcpp::FunctionTree<D> re;
    mrcpp::FunctionTree<D> im;

    explicit CplxFun(const mrcpp::MultiResolutionAnalysis<D> &mra_)
            : mra(mra_)
            , re(mra_)
            , im(mra_) {}

    CplxFun(const CplxFun &) = delete;
    CplxFun &operator=(const CplxFun &) = delete;
};

/** Replace dst with a deep copy of src (grid + MW coefficients). */
template <int D>
void copy_into(mrcpp::FunctionTree<D> &dst, mrcpp::FunctionTree<D> &src) {
    dst.clear();
    src.deep_copy(&dst);
}

template <int D>
void copy_into(CplxFun<D> &dst, CplxFun<D> &src) {
    copy_into(dst.re, src.re);
    copy_into(dst.im, src.im);
}

/** ||ψ||₂² = ∫ |ψ|² = ||Re||² + ||Im||² */
template <int D>
double square_norm(const CplxFun<D> &psi) {
    const double nre = psi.re.getSquareNorm();
    const double nim = psi.im.getSquareNorm();
    const double a = (nre > 0.0) ? nre : 0.0;
    const double b = (nim > 0.0) ? nim : 0.0;
    return a + b;
}

template <int D>
double norm(const CplxFun<D> &psi) {
    return std::sqrt(square_norm(psi));
}

template <int D>
void normalize(CplxFun<D> &psi) {
    const double n = norm(psi);
    if (n <= 0.0) {
        throw std::runtime_error("cannot normalize a zero wave function");
    }
    psi.re.rescale(1.0 / n);
    psi.im.rescale(1.0 / n);
}

/** Hermitian inner product ⟨a|b⟩ = ∫ a* b */
template <int D>
std::complex<double> inner(CplxFun<D> &a, CplxFun<D> &b) {
    const double rr = mrcpp::dot(a.re, b.re);
    const double ii = mrcpp::dot(a.im, b.im);
    const double ri = mrcpp::dot(a.re, b.im);
    const double ir = mrcpp::dot(a.im, b.re);
    return {rr + ii, ri - ir};
}

/**
 * out = Σ_k c_k inp_k   (real trees).
 * The output tree must be undefined (freshly constructed or clear()'d).
 */
template <int D>
void linear_combine(double prec,
                    mrcpp::FunctionTree<D> &out,
                    const std::vector<std::pair<double, mrcpp::FunctionTree<D> *>> &terms) {
    mrcpp::FunctionTreeVector<D> vec;
    vec.reserve(terms.size());
    for (auto &t : terms) {
        vec.push_back(std::make_tuple(t.first, t.second));
    }
    mrcpp::add(prec, out, vec);
}

/** Crop wavelet coefficients below `prec` (adaptive coarsening). */
template <int D>
void crop(CplxFun<D> &psi, double prec) {
    psi.re.crop(prec);
    psi.im.crop(prec);
}

} // namespace tdse

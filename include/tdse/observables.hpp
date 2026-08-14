#pragma once

#include "tdse/analytic.hpp"
#include "tdse/operators.hpp"
#include "tdse/parameters.hpp"
#include "tdse/wavefunction.hpp"

#include "MRCPP/MWFunctions"
#include "MRCPP/Printer"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace tdse {

struct Observables {
    double t = 0.0;
    double nrm = 0.0;
    double dipole = 0.0;
    double energy = 0.0;
    double overlap_analytic = 0.0; ///< |⟨ψ_num|ψ_ana⟩| for free / HO tests
    double dipole_analytic = std::numeric_limits<double>::quiet_NaN();
    double energy_analytic = std::numeric_limits<double>::quiet_NaN();
    int n_nodes_re = 0;
    int n_nodes_im = 0;
};

template <int D>
double dipole_moment(double prec, CplxFun<D> &psi, int axis = 0) {
    mrcpp::FunctionTree<D> re2(psi.mra);
    mrcpp::FunctionTree<D> im2(psi.mra);
    mrcpp::square(prec, re2, psi.re);
    mrcpp::square(prec, im2, psi.im);
    mrcpp::FunctionTree<D> dens(psi.mra);
    mrcpp::add(prec, dens, 1.0, re2, 1.0, im2);

    mrcpp::FunctionTree<D> x(psi.mra);
    CoordinateFunction<D> xf(axis);
    mrcpp::project<D, double>(prec, x, xf);

    mrcpp::FunctionTree<D> xdens(psi.mra);
    mrcpp::multiply(prec, xdens, 1.0, x, dens);
    return xdens.integrate();
}

/** Total dipole Σ_d ⟨x_d⟩ for exact 1D N-body, otherwise ⟨x⟩. */
template <int D>
double total_dipole(double prec, CplxFun<D> &psi, const Parameters &p) {
    const bool nbody_1d = (p.representation == Representation::Exact && p.spatial_dim == 1 && p.n_electrons > 1);
    if (!nbody_1d) {
        return dipole_moment<D>(prec, psi, 0);
    }
    double mu = 0.0;
    for (int d = 0; d < D; ++d) {
        mu += dipole_moment<D>(prec, psi, d);
    }
    return mu;
}

template <int D>
double energy_expectation(double prec, CplxFun<D> &psi, OperatorSet<D> &ops, mrcpp::FunctionTree<D> &V) {
    CplxFun<D> Hpsi(psi.mra);
    apply_hamiltonian(prec, Hpsi, ops, psi, V);
    return inner(psi, Hpsi).real();
}

template <int D>
Observables compute_observables(double prec,
                                double t,
                                CplxFun<D> &psi,
                                OperatorSet<D> &ops,
                                mrcpp::FunctionTree<D> &V,
                                const Parameters &p) {
    Observables o;
    o.t = t;
    o.nrm = norm(psi);
    o.dipole = total_dipole<D>(prec, psi, p);
    o.energy = energy_expectation(prec, psi, ops, V);
    o.n_nodes_re = psi.re.getNNodes();
    o.n_nodes_im = psi.im.getNNodes();
    return o;
}

inline void write_header(std::ostream &os) {
    os << "t,norm,dipole,energy,nodes_re,nodes_im,overlap_analytic,dipole_analytic,energy_analytic\n";
}

inline void write_row(std::ostream &os, const Observables &o) {
    os << std::scientific << std::setprecision(12)
       << o.t << ',' << o.nrm << ',' << o.dipole << ',' << o.energy << ','
       << o.n_nodes_re << ',' << o.n_nodes_im << ',' << o.overlap_analytic << ','
       << o.dipole_analytic << ',' << o.energy_analytic << '\n';
}

inline void print_row(const Observables &o) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(6)
       << " t=" << o.t
       << "  ||ψ||=" << std::setprecision(10) << o.nrm
       << "  μ=" << o.dipole
       << "  E=" << o.energy
       << "  nodes=" << o.n_nodes_re << "/" << o.n_nodes_im;
    if (o.overlap_analytic > 0.0) {
        ss << "  |⟨num|ana⟩|=" << std::setprecision(10) << o.overlap_analytic;
    }
    if (std::isfinite(o.dipole_analytic)) {
        ss << "  μ_ana=" << std::setprecision(6) << o.dipole_analytic;
    }
    if (std::isfinite(o.energy_analytic)) {
        ss << "  E_ana=" << std::setprecision(6) << o.energy_analytic;
    }
    println(0, ss.str());
}

} // namespace tdse

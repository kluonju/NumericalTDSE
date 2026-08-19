#pragma once

/**
 * Interacting density-to-potential inversion for two electrons
 * (Thiele, Gross, Kümmel, PRL 100, 153004 (2008); Peirs, Van Neck, Waroquier,
 * Phys. Rev. A 67, 012505 (2003)).
 *
 * Given a target one-body density n*(r), find the local v_ext(r) whose
 * interacting singlet ground state reproduces n*. The update is
 *
 *   v^{(i)}(r) = v^{(i−1)}(r) + γ (w0 + |r|^β) (n^{(i−1)}(r) − n*(r))
 *
 * After v_ext is known, the adiabatically exact KS potential of the two-electron
 * singlet is algebraic, φ = sqrt(n/2),
 *
 *   v_s = (1/(2φ)) ∇²φ + const,    v_c = v_s − ½ v_H − v_ext.
 *
 * Representation. Two electrons in d spatial dimensions live in 2d-dimensional
 * configuration space. For dim=2 that is the default: ψ(x1,y1,x2,y2) on an N⁴
 * grid (exact 2e in 2D, stored as 1e in 4D). Set INVERT basis='orbital' for a
 * truncated 2e-in-2D CI in a 2D HO orbital basis on the same spatial grid;
 * v_ext(x,y) and n(x,y) stay two-dimensional in both cases.
 */

#include "tdse/invert_ci.hpp"
#include "tdse/nbody_grid.hpp"
#include "tdse/parameters.hpp"
#include "tdse/parallel.hpp"

#include "MRCPP/Printer"
#include "MRCPP/Timer"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tdse {

inline double spatial_radius(const TwoElectronGrid<1> &g, std::size_t i) {
    return std::abs(g.coord(static_cast<int>(i)));
}

inline double spatial_radius(const TwoElectronGrid<2> &g, std::size_t u) {
    const int i = static_cast<int>(u % static_cast<std::size_t>(g.n));
    const int j = static_cast<int>(u / static_cast<std::size_t>(g.n));
    return std::sqrt(g.radius2(i, j));
}

inline void shift_to_match(std::vector<double> &v, const std::vector<double> &ref, int i0) {
    if (i0 < 0 || static_cast<std::size_t>(i0) >= v.size() || static_cast<std::size_t>(i0) >= ref.size()) {
        return;
    }
    if (!std::isfinite(ref[static_cast<std::size_t>(i0)]) || !std::isfinite(v[static_cast<std::size_t>(i0)])) {
        return;
    }
    const double d = ref[static_cast<std::size_t>(i0)] - v[static_cast<std::size_t>(i0)];
    for (double &x : v) {
        x += d;
    }
}

inline double weighted_v_rms(const std::vector<double> &a,
                             const std::vector<double> &b,
                             const std::vector<double> &n,
                             double ncut,
                             double dV) {
    double num = 0.0;
    double den = 0.0;
    const std::size_t m = a.size();
    for (std::size_t i = 0; i < m; ++i) {
        if (n[i] < ncut || !std::isfinite(a[i]) || !std::isfinite(b[i])) {
            continue;
        }
        const double dv = a[i] - b[i];
        num += n[i] * dv * dv;
        den += n[i];
    }
    if (den <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    (void)dV;
    return std::sqrt(num / den);
}

template <int SpatialDim>
void apply_guess(TwoElectronGrid<SpatialDim> &g,
                 const Parameters &p,
                 const std::vector<double> &v_true,
                 const std::vector<double> &n_target) {
    switch (p.invert_guess) {
        case InvertGuess::Zero:
            std::fill(g.v_ext.begin(), g.v_ext.end(), 0.0);
            break;
        case InvertGuess::Harmonic:
            g.fill_harmonic(g.v_ext, p.omega);
            break;
        case InvertGuess::Atom:
            g.fill_atom(g.v_ext, p.Z);
            break;
        case InvertGuess::Scaled: {
            const bool have_true = !v_true.empty() && std::isfinite(v_true[v_true.size() / 2]);
            if (have_true) {
                g.v_ext = v_true;
            } else {
                g.fill_harmonic(g.v_ext, p.omega);
            }
            for (double &x : g.v_ext) {
                x *= p.invert_scale;
            }
            break;
        }
        case InvertGuess::Hx: {
            g.dens = n_target;
            std::vector<double> vs;
            std::vector<double> vh;
            g.ks_potential(vs, p.invert_ncut);
            g.hartree(vh);
            g.v_ext.resize(g.ns);
            for (std::size_t i = 0; i < g.ns; ++i) {
                g.v_ext[i] = vs[i] - 0.5 * vh[i];
            }
            break;
        }
    }
}

template <int SpatialDim>
void load_density_file(TwoElectronGrid<SpatialDim> &g, const std::string &path, std::vector<double> &n_target) {
    std::ifstream in(path);
    if (!in) {
        throw std::invalid_argument("cannot open invert density file: " + path);
    }
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> nsamp;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '!') {
            continue;
        }
        std::istringstream iss(line);
        if (line.find(',') != std::string::npos) {
            for (char &c : line) {
                if (c == ',') {
                    c = ' ';
                }
            }
            iss.str(line);
            iss.clear();
        }
        double x = 0.0;
        double y = 0.0;
        double val = 0.0;
        if constexpr (SpatialDim == 1) {
            if (!(iss >> x >> val)) {
                continue;
            }
            xs.push_back(x);
            nsamp.push_back(val);
        } else {
            if (!(iss >> x >> y >> val)) {
                continue;
            }
            xs.push_back(x);
            ys.push_back(y);
            nsamp.push_back(val);
        }
    }
    if (nsamp.empty()) {
        throw std::invalid_argument("invert density file is empty: " + path);
    }
    n_target.assign(g.ns, 0.0);
    auto nearest = [&](double xq, double yq) -> double {
        double best = 0.0;
        double d2m = std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < nsamp.size(); ++k) {
            const double dx = xs[k] - xq;
            double d2 = dx * dx;
            if constexpr (SpatialDim == 2) {
                const double dy = ys[k] - yq;
                d2 += dy * dy;
            }
            if (d2 < d2m) {
                d2m = d2;
                best = nsamp[k];
            }
        }
        return best;
    };
    if constexpr (SpatialDim == 1) {
        for (int i = 0; i < g.n; ++i) {
            n_target[g.spat(i)] = nearest(g.coord(i), 0.0);
        }
    } else {
        for (int j = 0; j < g.n; ++j) {
            for (int i = 0; i < g.n; ++i) {
                n_target[g.spat(i, j)] = nearest(g.coord(i), g.coord(j));
            }
        }
    }
}

template <int SpatialDim>
void write_density_table(const Parameters &p,
                         const TwoElectronGrid<SpatialDim> &g,
                         const std::vector<double> &n_target,
                         const std::vector<double> &v_true,
                         const std::vector<double> &v_inv,
                         const std::vector<double> &vs,
                         const std::vector<double> &vh,
                         const std::vector<double> &vc) {
    if (!parallel::io_rank()) {
        return;
    }
    std::string path = p.prefix.empty() ? "invert_density.csv" : (p.prefix + "_density.csv");
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
    out << std::scientific << std::setprecision(12);
    if constexpr (SpatialDim == 1) {
        out << "x,n_target,n,v_true,v_inv,v_s,v_h,v_c\n";
        for (int i = 0; i < g.n; ++i) {
            const std::size_t u = g.spat(i);
            out << g.coord(i) << ',' << n_target[u] << ',' << g.dens[u] << ',' << v_true[u] << ',' << v_inv[u] << ','
                << vs[u] << ',' << vh[u] << ',' << vc[u] << '\n';
        }
    } else {
        out << "x,y,n_target,n,v_true,v_inv,v_s,v_h,v_c\n";
        for (int j = 0; j < g.n; ++j) {
            for (int i = 0; i < g.n; ++i) {
                const std::size_t u = g.spat(i, j);
                out << g.coord(i) << ',' << g.coord(j) << ',' << n_target[u] << ',' << g.dens[u] << ',' << v_true[u]
                    << ',' << v_inv[u] << ',' << vs[u] << ',' << vh[u] << ',' << vc[u] << '\n';
            }
        }
    }
    println(0, "  wrote density/potential table '" << path << "'");
}

template <int SpatialDim>
int invert_two_electrons(const Parameters &p) {
    mrcpp::Timer timer;
    TwoElectronGrid<SpatialDim> g;
    g.setup(p);

    mrcpp::print::header(0, "Two-electron density inversion (TGK08)");
    println(0,
            "  representation  : 2 electrons in " << SpatialDim << "D ("
                                                  << TwoElectronGrid<SpatialDim>::kConfigDim
                                                  << "D configuration-space grid)");
    println(0, "  spatial dim     : " << SpatialDim);
    mrcpp::print::value(0, "n_grid", static_cast<double>(g.n));
    mrcpp::print::value(0, "L", g.L);
    mrcpp::print::value(0, "dx", g.dx);
    mrcpp::print::value(0, "config size", static_cast<double>(g.nc));
    println(0, "  target          : " << invert_target_name(p.invert_target));
    println(0, "  guess           : " << invert_guess_name(p.invert_guess));
    println(0, "  ee              : " << (g.ee ? "true" : "false"));

    if (parallel::size > 1) {
        println(0, "  note: inversion is OpenMP-only; extra MPI ranks replicate the work");
    }

    std::vector<double> v_true(g.ns, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> n_target(g.ns, 0.0);

    g.gaussian_trial(p.alpha > 0.0 ? p.alpha : 1.0);

    if (p.invert_target == InvertTarget::File) {
        load_density_file(g, p.invert_density_file, n_target);
        g.fill_trap(v_true, p);
    } else {
        g.fill_trap(v_true, p);
        g.v_ext = v_true;
        g.build_config_potential();
        const int warm = std::max(p.invert_inner * 2, 80);
        const double E0 = g.relax(warm, p.invert_tau, p.eigen_thr, p.print_every, "target GS relax");
        g.compute_density();
        n_target = g.dens;
        println(0,
                "  target GS       : E = " << std::setprecision(10) << E0 << "  N = " << g.density_integral()
                                           << "  ||(H-E)ψ|| = " << g.residual(E0));
    }

    apply_guess(g, p, v_true, n_target);
    g.build_config_potential();
    const int warm_guess = std::max(p.invert_inner, 40);
    double E = g.relax(warm_guess, p.invert_tau, p.eigen_thr, p.print_every, "guess relax");
    g.compute_density();
    const double l1_init = g.l1_density(n_target);
    println(0, "  initial L1      : ∫|n−n*| = " << std::setprecision(8) << l1_init);

    std::vector<double> vs;
    std::vector<double> vh;
    std::vector<double> vc(g.ns, 0.0);
    g.ks_potential(vs, p.invert_ncut);
    g.hartree(vh);
    const int i0 = g.density_peak();
    shift_to_match(vs, v_true, i0);

    if (p.invert_ks_only) {
        for (std::size_t i = 0; i < g.ns; ++i) {
            vc[i] = vs[i] - 0.5 * vh[i] - v_true[i];
        }
        write_density_table(p, g, n_target, v_true, g.v_ext, vs, vh, vc);
        mrcpp::print::footer(0, timer, 2);
        return 0;
    }

    std::ofstream hist;
    if (parallel::io_rank()) {
        hist.open(p.output);
        if (!hist) {
            throw std::runtime_error("cannot write " + p.output);
        }
        hist << "iter,l1,energy,gamma,v_rms,n_int,residual\n";
        hist << std::scientific << std::setprecision(12);
    }

    double gamma = p.invert_gamma;
    double l1 = l1_init;
    std::vector<double> v_save = g.v_ext;
    std::vector<double> psi_save = g.psi;

    auto v_rms_now = [&]() {
        std::vector<double> vcmp = g.v_ext;
        shift_to_match(vcmp, v_true, i0);
        return weighted_v_rms(vcmp, v_true, n_target, p.invert_ncut, g.dV_s);
    };

    if (parallel::io_rank()) {
        hist << 0 << ',' << l1 << ',' << E << ',' << gamma << ',' << v_rms_now() << ',' << g.density_integral() << ','
             << g.residual(E) << '\n';
    }

    mrcpp::print::header(0, "Peirs / TGK08 iteration");
    println(0,
            "  maxiter=" << p.invert_maxiter << "  inner=" << p.invert_inner << "  print_every="
                       << p.print_every);
    for (int it = 1; it <= p.invert_maxiter; ++it) {
        v_save = g.v_ext;
        psi_save = g.psi;

        for (std::size_t u = 0; u < g.ns; ++u) {
            const double r = spatial_radius(g, u);
            const double w = gamma * (p.invert_w0 + std::pow(r, p.invert_beta));
            double dv = w * (g.dens[u] - n_target[u]);
            if (dv > p.invert_dvmax) {
                dv = p.invert_dvmax;
            } else if (dv < -p.invert_dvmax) {
                dv = -p.invert_dvmax;
            }
            g.v_ext[u] += dv;
        }
        g.build_config_potential();
        println(0, "  iter " << it << "  inner relax (" << p.invert_inner << " imag-time steps):");
        E = g.relax(p.invert_inner, p.invert_tau, p.eigen_thr, p.print_every, "inner");
        g.compute_density();
        const double l1_new = g.l1_density(n_target);
        if (l1_new > l1 * 1.05 && gamma > 1.0e-5) {
            g.v_ext = v_save;
            g.psi = psi_save;
            g.build_config_potential();
            g.compute_density();
            gamma *= 0.5;
            println(0, "  iter " << it << "  backtrack, gamma → " << gamma);
            continue;
        }
        if (l1_new < l1) {
            gamma = std::min(p.invert_gamma, gamma * 1.05);
        }
        l1 = l1_new;
        const double vrms = v_rms_now();
        const double nint = g.density_integral();
        const double res = g.residual(E);
        println(0,
                "  iter " << it << "  L1=" << std::setprecision(6) << l1 << "  E=" << std::setprecision(8) << E
                          << "  γ=" << gamma << "  v_rms=" << vrms << "  N=" << nint);
        if (parallel::io_rank()) {
            hist << it << ',' << l1 << ',' << E << ',' << gamma << ',' << vrms << ',' << nint << ',' << res << '\n';
            hist.flush();
        }
        if (l1 < p.invert_tol) {
            println(0, "  converged: ∫|n−n*| < " << p.invert_tol);
            break;
        }
    }

    std::vector<double> v_inv = g.v_ext;
    g.ks_potential(vs, p.invert_ncut);
    g.hartree(vh);
    shift_to_match(v_inv, v_true, i0);
    shift_to_match(vs, v_true, i0);
    for (std::size_t i = 0; i < g.ns; ++i) {
        vc[i] = vs[i] - 0.5 * vh[i] - v_inv[i];
    }
    const double vrms = weighted_v_rms(v_inv, v_true, n_target, p.invert_ncut, g.dV_s);
    println(0, "  final L1        : " << std::setprecision(8) << l1);
    println(0, "  final v_rms     : " << vrms << "  (density-weighted, n > ncut, constants aligned)");
    println(0, "  particle number : " << g.density_integral() << "  (should be 2)");

    write_density_table(p, g, n_target, v_true, v_inv, vs, vh, vc);
    if (hist.is_open()) {
        hist.close();
        println(0, "  wrote inversion history '" << p.output << "'");
    }
    mrcpp::print::footer(0, timer, 2);

    if (p.invert_check) {
        const bool improved = (l1 < l1_init * 0.8) || (l1 < p.invert_tol);
        if (!improved) {
            std::cerr << "NumericalTDSE error: inversion L1 did not improve (" << l1_init << " → " << l1 << ")\n";
            return 1;
        }
    }
    return 0;
}

/** 2e-in-2D inversion: singlet CI in a 2D HO orbital basis (not 1e in 4D). */
inline int invert_two_electrons_ci(const Parameters &p) {
    mrcpp::Timer timer;
    TwoElectronGrid<2> g;
    g.setup(p);

    TwoElectronCI2D ci;
    println(0, "  building 2D HO orbital CI (n_orb=" << p.invert_norb << ", n_grid=" << g.n << ")...");
    ci.setup(g, p);

    mrcpp::print::header(0, "Two-electron density inversion (TGK08), 2e-in-2D CI");
    println(0, "  representation  : 2 electrons in 2D, singlet CI in " << ci.M << " HO orbitals");
    println(0, "  spatial dim     : 2");
    mrcpp::print::value(0, "n_grid", static_cast<double>(g.n));
    mrcpp::print::value(0, "n_orb", static_cast<double>(ci.M));
    mrcpp::print::value(0, "CI dimension", static_cast<double>(ci.M * ci.M));
    mrcpp::print::value(0, "L", g.L);
    mrcpp::print::value(0, "dx", g.dx);
    println(0, "  target          : " << invert_target_name(p.invert_target));
    println(0, "  guess           : " << invert_guess_name(p.invert_guess));
    println(0, "  ee              : " << (g.ee ? "true" : "false"));
    println(0, "  inner solver    : full CI diagonalization (INVERT inner/tau unused)");

    if (parallel::size > 1) {
        println(0, "  note: inversion is OpenMP-only; extra MPI ranks replicate the work");
    }

    std::vector<double> v_true(g.ns, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> n_target(g.ns, 0.0);

    if (p.invert_target == InvertTarget::File) {
        load_density_file(g, p.invert_density_file, n_target);
        g.fill_trap(v_true, p);
    } else {
        g.fill_trap(v_true, p);
        g.v_ext = v_true;
        const double E0 = ci.ground(g.v_ext);
        n_target = g.dens;
        println(0,
                "  target GS       : E = " << std::setprecision(10) << E0 << "  N = " << g.density_integral()
                                           << "  ||(H-E)C|| = " << ci.residual);
    }

    apply_guess(g, p, v_true, n_target);
    double E = ci.ground(g.v_ext);
    const double l1_init = g.l1_density(n_target);
    println(0, "  initial L1      : ∫|n−n*| = " << std::setprecision(8) << l1_init);

    std::vector<double> vs;
    std::vector<double> vh;
    std::vector<double> vc(g.ns, 0.0);
    g.ks_potential(vs, p.invert_ncut);
    g.hartree(vh);
    const int i0 = g.density_peak();
    shift_to_match(vs, v_true, i0);

    if (p.invert_ks_only) {
        for (std::size_t i = 0; i < g.ns; ++i) {
            vc[i] = vs[i] - 0.5 * vh[i] - v_true[i];
        }
        write_density_table(p, g, n_target, v_true, g.v_ext, vs, vh, vc);
        mrcpp::print::footer(0, timer, 2);
        return 0;
    }

    std::ofstream hist;
    if (parallel::io_rank()) {
        hist.open(p.output);
        if (!hist) {
            throw std::runtime_error("cannot write " + p.output);
        }
        hist << "iter,l1,energy,gamma,v_rms,n_int,residual\n";
        hist << std::scientific << std::setprecision(12);
    }

    double gamma = p.invert_gamma;
    double l1 = l1_init;
    std::vector<double> v_save = g.v_ext;

    auto v_rms_now = [&]() {
        std::vector<double> vcmp = g.v_ext;
        shift_to_match(vcmp, v_true, i0);
        return weighted_v_rms(vcmp, v_true, n_target, p.invert_ncut, g.dV_s);
    };

    if (parallel::io_rank()) {
        hist << 0 << ',' << l1 << ',' << E << ',' << gamma << ',' << v_rms_now() << ',' << g.density_integral() << ','
             << ci.residual << '\n';
    }

    mrcpp::print::header(0, "Peirs / TGK08 iteration");
    println(0, "  maxiter=" << p.invert_maxiter << "  print_every=" << p.print_every);
    for (int it = 1; it <= p.invert_maxiter; ++it) {
        v_save = g.v_ext;
        for (std::size_t u = 0; u < g.ns; ++u) {
            const double r = spatial_radius(g, u);
            const double w = gamma * (p.invert_w0 + std::pow(r, p.invert_beta));
            double dv = w * (g.dens[u] - n_target[u]);
            if (dv > p.invert_dvmax) {
                dv = p.invert_dvmax;
            } else if (dv < -p.invert_dvmax) {
                dv = -p.invert_dvmax;
            }
            g.v_ext[u] += dv;
        }
        E = ci.ground(g.v_ext);
        const double l1_new = g.l1_density(n_target);
        if (l1_new > l1 * 1.05 && gamma > 1.0e-5) {
            g.v_ext = v_save;
            E = ci.ground(g.v_ext);
            gamma *= 0.5;
            println(0, "  iter " << it << "  backtrack, gamma → " << gamma);
            continue;
        }
        if (l1_new < l1) {
            gamma = std::min(p.invert_gamma, gamma * 1.05);
        }
        l1 = l1_new;
        const double vrms = v_rms_now();
        const double nint = g.density_integral();
        println(0,
                "  iter " << it << "  L1=" << std::setprecision(6) << l1 << "  E=" << std::setprecision(8) << E
                          << "  γ=" << gamma << "  v_rms=" << vrms << "  N=" << nint
                          << "  ||(H-E)C||=" << ci.residual);
        if (parallel::io_rank()) {
            hist << it << ',' << l1 << ',' << E << ',' << gamma << ',' << vrms << ',' << nint << ',' << ci.residual
                 << '\n';
            hist.flush();
        }
        if (l1 < p.invert_tol) {
            println(0, "  converged: ∫|n−n*| < " << p.invert_tol);
            break;
        }
    }

    std::vector<double> v_inv = g.v_ext;
    g.ks_potential(vs, p.invert_ncut);
    g.hartree(vh);
    shift_to_match(v_inv, v_true, i0);
    shift_to_match(vs, v_true, i0);
    for (std::size_t i = 0; i < g.ns; ++i) {
        vc[i] = vs[i] - 0.5 * vh[i] - v_inv[i];
    }
    const double vrms = weighted_v_rms(v_inv, v_true, n_target, p.invert_ncut, g.dV_s);
    println(0, "  final L1        : " << std::setprecision(8) << l1);
    println(0, "  final v_rms     : " << vrms << "  (density-weighted, n > ncut, constants aligned)");
    println(0, "  particle number : " << g.density_integral() << "  (should be 2)");

    write_density_table(p, g, n_target, v_true, v_inv, vs, vh, vc);
    if (hist.is_open()) {
        hist.close();
        println(0, "  wrote inversion history '" << p.output << "'");
    }
    mrcpp::print::footer(0, timer, 2);

    if (p.invert_check) {
        const bool improved = (l1 < l1_init * 0.8) || (l1 < p.invert_tol);
        if (!improved) {
            std::cerr << "NumericalTDSE error: inversion L1 did not improve (" << l1_init << " → " << l1 << ")\n";
            return 1;
        }
    }
    return 0;
}

inline int run_invert(const Parameters &p) {
    if (p.n_electrons != 2) {
        throw std::invalid_argument("calculation = 'invert' requires SYSTEM electrons = 2");
    }
    if (p.spatial_dim == 1) {
        if (p.invert_basis == InvertBasis::Orbital) {
            throw std::invalid_argument("INVERT basis = 'orbital' is 2e-in-2D CI; for dim=1 use the N×N config grid");
        }
        return invert_two_electrons<1>(p);
    }
    if (p.spatial_dim == 2) {
        if (p.invert_basis == InvertBasis::Orbital) {
            return invert_two_electrons_ci(p);
        }
        return invert_two_electrons<2>(p);
    }
    throw std::invalid_argument("invert supports spatial dim 1 (TGK08) or 2 (config grid or orbital CI)");
}

} // namespace tdse

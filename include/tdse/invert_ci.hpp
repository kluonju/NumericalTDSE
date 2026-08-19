#pragma once

/**
 * Two-electron-in-2D inversion (CI in a 2D orbital basis).
 *
 * Distinct from the 4D configuration-space grid (one particle in 4D):
 * each electron is expanded in 2D harmonic-oscillator orbitals φ_a(x,y).
 * The singlet ground state is the lowest even eigenvector of the two-body
 * Hamiltonian on the M×M product basis. Two-electron integrals are built
 * once; each Peirs step only updates the one-body v_ext(x,y).
 */

#include "tdse/analytic.hpp"
#include "tdse/nbody_grid.hpp"
#include "tdse/parameters.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tdse {

struct TwoElectronCI2D {
    TwoElectronGrid<2> *g = nullptr;
    int M = 0;
    double omega_b = 1.0;
    std::vector<std::vector<double>> phi; ///< M orbitals on the spatial grid
    Eigen::MatrixXd hop_T;                ///< kinetic + (will add v each solve)
    Eigen::MatrixXd Wmat;                 ///< ⟨pq|W|rs⟩ at (p+M*q, r+M*s)
    Eigen::MatrixXd C;                    ///< ground C_{pq}, ||C||_F = 1
    double energy = 0.0;
    double residual = 0.0;

    /** Product-state index matching column-major C(p,q). */
    int idx(int p, int q) const { return p + M * q; }

    void setup(TwoElectronGrid<2> &grid, const Parameters &p) {
        g = &grid;
        M = std::max(3, p.invert_norb);
        omega_b = (p.omega > 0.0) ? p.omega : 1.0;
        build_orbitals();
        build_kinetic();
        build_interaction();
    }

    void build_orbitals() {
        const int n = g->n;
        const std::size_t ns = g->ns;
        std::vector<std::pair<int, int>> qn;
        for (int shell = 0; static_cast<int>(qn.size()) < M; ++shell) {
            for (int nx = 0; nx <= shell; ++nx) {
                qn.emplace_back(nx, shell - nx);
            }
        }
        qn.resize(static_cast<std::size_t>(M));

        Eigen::MatrixXd raw(static_cast<int>(ns), M);
        for (int a = 0; a < M; ++a) {
            const int nx = qn[static_cast<std::size_t>(a)].first;
            const int ny = qn[static_cast<std::size_t>(a)].second;
            for (int j = 0; j < n; ++j) {
                const double py = ho_eigen_1d(g->coord(j), ny, omega_b).real();
                for (int i = 0; i < n; ++i) {
                    const double px = ho_eigen_1d(g->coord(i), nx, omega_b).real();
                    raw(static_cast<int>(g->spat(i, j)), a) = px * py;
                }
            }
            // Kill Dirichlet rim so the kinetic stencil is consistent.
            for (int j = 0; j < n; ++j) {
                raw(static_cast<int>(g->spat(0, j)), a) = 0.0;
                raw(static_cast<int>(g->spat(n - 1, j)), a) = 0.0;
            }
            for (int i = 0; i < n; ++i) {
                raw(static_cast<int>(g->spat(i, 0)), a) = 0.0;
                raw(static_cast<int>(g->spat(i, n - 1)), a) = 0.0;
            }
        }

        Eigen::MatrixXd S = raw.transpose() * raw * g->dV_s;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
        Eigen::VectorXd invsqrt(M);
        for (int i = 0; i < M; ++i) {
            const double lam = std::max(es.eigenvalues()(i), 1.0e-16);
            invsqrt(i) = 1.0 / std::sqrt(lam);
        }
        const Eigen::MatrixXd Sinvh = es.eigenvectors() * invsqrt.asDiagonal() * es.eigenvectors().transpose();
        const Eigen::MatrixXd ortho = raw * Sinvh;

        phi.assign(static_cast<std::size_t>(M), std::vector<double>(ns, 0.0));
        for (int a = 0; a < M; ++a) {
            for (std::size_t u = 0; u < ns; ++u) {
                phi[static_cast<std::size_t>(a)][u] = ortho(static_cast<int>(u), a);
            }
        }
    }

    void laplacian_2d(const std::vector<double> &f, std::vector<double> &out) const {
        const int n = g->n;
        const double c = 1.0 / (g->dx * g->dx);
        out.assign(g->ns, 0.0);
        for (int j = 1; j < n - 1; ++j) {
            for (int i = 1; i < n - 1; ++i) {
                const std::size_t u = g->spat(i, j);
                out[u] = c * (f[g->spat(i + 1, j)] + f[g->spat(i - 1, j)] + f[g->spat(i, j + 1)] +
                              f[g->spat(i, j - 1)] - 4.0 * f[u]);
            }
        }
    }

    void build_kinetic() {
        hop_T = Eigen::MatrixXd::Zero(M, M);
        std::vector<double> lap;
        for (int r = 0; r < M; ++r) {
            laplacian_2d(phi[static_cast<std::size_t>(r)], lap);
            for (int p = 0; p < M; ++p) {
                double s = 0.0;
                for (std::size_t u = 0; u < g->ns; ++u) {
                    s += phi[static_cast<std::size_t>(p)][u] * (-0.5 * lap[u]);
                }
                hop_T(p, r) = s * g->dV_s;
            }
        }
        hop_T = 0.5 * (hop_T + hop_T.transpose());
    }

    void build_interaction() {
        const int n = g->n;
        const std::size_t ns = g->ns;
        const int MM = M * M;
        Wmat = Eigen::MatrixXd::Zero(MM, MM);
        if (!g->ee) {
            return;
        }
        std::vector<double> rho(ns, 0.0);
        std::vector<double> pot(ns, 0.0);
        for (int q = 0; q < M; ++q) {
            for (int s = 0; s <= q; ++s) {
                const auto &fq = phi[static_cast<std::size_t>(q)];
                const auto &fs = phi[static_cast<std::size_t>(s)];
                for (std::size_t v = 0; v < ns; ++v) {
                    rho[v] = fq[v] * fs[v];
                }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < n; ++i) {
                        double acc = 0.0;
                        const double xi = g->coord(i);
                        const double yi = g->coord(j);
                        for (int l = 0; l < n; ++l) {
                            const double yl = g->coord(l);
                            for (int k = 0; k < n; ++k) {
                                const double dx = xi - g->coord(k);
                                const double dy = yi - yl;
                                acc += rho[g->spat(k, l)] * g->soft_w(dx * dx + dy * dy);
                            }
                        }
                        pot[g->spat(i, j)] = acc * g->dV_s;
                    }
                }
                for (int p = 0; p < M; ++p) {
                    for (int r = 0; r <= p; ++r) {
                        double acc = 0.0;
                        const auto &fp = phi[static_cast<std::size_t>(p)];
                        const auto &fr = phi[static_cast<std::size_t>(r)];
                        for (std::size_t u = 0; u < ns; ++u) {
                            acc += fp[u] * fr[u] * pot[u];
                        }
                        acc *= g->dV_s;
                        auto put = [&](int a, int b, int c, int d) {
                            Wmat(idx(a, b), idx(c, d)) = acc;
                        };
                        // ⟨pq|W|rs⟩ = I_{pr;qs} and real-orbital / electron-swap copies
                        put(p, q, r, s);
                        put(r, q, p, s);
                        put(p, s, r, q);
                        put(r, s, p, q);
                        put(q, p, s, r);
                        put(q, r, s, p);
                        put(s, p, q, r);
                        put(s, r, q, p);
                    }
                }
            }
        }
        Wmat = 0.5 * (Wmat + Wmat.transpose());
    }

    Eigen::MatrixXd one_body(const std::vector<double> &vext) const {
        Eigen::MatrixXd h = hop_T;
        for (int p = 0; p < M; ++p) {
            for (int r = 0; r <= p; ++r) {
                double s = 0.0;
                const auto &fp = phi[static_cast<std::size_t>(p)];
                const auto &fr = phi[static_cast<std::size_t>(r)];
                for (std::size_t u = 0; u < g->ns; ++u) {
                    s += fp[u] * vext[u] * fr[u];
                }
                s *= g->dV_s;
                h(p, r) += s;
                if (p != r) {
                    h(r, p) += s;
                }
            }
        }
        return h;
    }

    double ground(const std::vector<double> &vext) {
        const Eigen::MatrixXd h = one_body(vext);
        const int MM = M * M;
        Eigen::MatrixXd H = Wmat;
        for (int p = 0; p < M; ++p) {
            for (int q = 0; q < M; ++q) {
                for (int r = 0; r < M; ++r) {
                    for (int s = 0; s < M; ++s) {
                        double val = 0.0;
                        if (p == r) {
                            val += h(q, s);
                        }
                        if (q == s) {
                            val += h(p, r);
                        }
                        H(idx(p, q), idx(r, s)) += val;
                    }
                }
            }
        }
        H = 0.5 * (H + H.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
        int pick = 0;
        double best_even = -1.0;
        for (int k = 0; k < std::min(8, MM); ++k) {
            Eigen::Map<const Eigen::MatrixXd> Ck(es.eigenvectors().col(k).data(), M, M);
            const double even = (Ck + Ck.transpose()).norm();
            const double odd = (Ck - Ck.transpose()).norm();
            const double score = even / (even + odd + 1.0e-16);
            if (score > best_even) {
                best_even = score;
                pick = k;
            }
            if (score > 0.95) {
                pick = k;
                break;
            }
        }
        Eigen::Map<const Eigen::MatrixXd> Craw(es.eigenvectors().col(pick).data(), M, M);
        C = 0.5 * (Craw + Craw.transpose());
        const double nrm = C.norm();
        if (nrm <= 0.0) {
            throw std::runtime_error("2e-2D CI: vanishing singlet ground state");
        }
        C /= nrm;
        Eigen::Map<const Eigen::VectorXd> cv(C.data(), MM);
        const Eigen::VectorXd Hc = H * cv;
        energy = cv.dot(Hc);
        residual = (Hc - energy * cv).norm();
        fill_density();
        return energy;
    }

    void fill_density() {
        const Eigen::MatrixXd D = C * C.transpose();
        g->dens.assign(g->ns, 0.0);
        for (int p = 0; p < M; ++p) {
            for (int r = 0; r < M; ++r) {
                const double dpr = 2.0 * D(p, r);
                if (std::abs(dpr) < 1.0e-18) {
                    continue;
                }
                const auto &fp = phi[static_cast<std::size_t>(p)];
                const auto &fr = phi[static_cast<std::size_t>(r)];
                for (std::size_t u = 0; u < g->ns; ++u) {
                    g->dens[u] += dpr * fp[u] * fr[u];
                }
            }
        }
    }
};

} // namespace tdse

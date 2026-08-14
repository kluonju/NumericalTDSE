#pragma once

#include <string>

namespace tdse {

enum class Propagator {
    Split,  ///< Strang splitting: exp(-iV dt/2) exp(-iT dt) exp(-iV dt/2)
    Krylov, ///< Short iterative Lanczos for exp(-i H dt)
    RK4     ///< Classical fourth-order Runge–Kutta on i ∂t ψ = H ψ
};

enum class KineticKind {
    ABGV, ///< Recommended MW derivative (Alpert–Beylkin–Ginez–Vozovoi)
    BS,   ///< B-spline / Anderson derivative, better for very smooth functions
    DConv ///< DerivativeConvolution (Gaussian approximation of δ')
};

enum class Representation {
    Exact,  ///< Full N-body wave function on FunctionTree<D>, D = n_e * dim ≤ 3
    Orbital ///< Up to 4 orbitals, each a FunctionTree<spatial_dim>
};

enum class TrapKind {
    Harmonic, ///< ½ ω² r²
    None,     ///< Free particle (plus optional laser)
    SoftAtom  ///< −Z / sqrt(r² + a²)
};

/**
 * All user-tunable numerical and physical parameters.
 *
 * Spatial MRA:
 *   The computational box is [-L, L]^D with interpolating (or Legendre)
 *   multiwavelets of polynomial order `order`, refined adaptively until the
 *   local wavelet-norm error is below `prec`, but never deeper than `max_depth`.
 */
struct Parameters {
    // --- Multiwavelet / MRA ---
    double prec = 1.0e-4;   ///< Adaptive refinement threshold (relative)
    int order = 7;          ///< Scaling-function polynomial order (k)
    int max_depth = 20;     ///< Maximum refinement levels relative to root
    double L = 8.0;         ///< Half-width of the world box, domain [-L, L]
    bool use_legendre = false; ///< Legendre basis (required for TimeEvolutionOperator)

    // --- Time integration ---
    double dt = 0.02;       ///< Time step Δt (atomic units)
    double T = 0.40;        ///< Final time
    int print_every = 1;    ///< Write observables every N steps
    Propagator propagator = Propagator::Krylov;
    KineticKind kinetic = KineticKind::ABGV;
    int krylov_dim = 12;    ///< Krylov subspace dimension for SIL / kinetic exp
    int teo_finest_scale = 8; ///< Uniform scale for TimeEvolutionOperator
    int teo_jpower = 20;    ///< Number of J-power integrals in the semigroup

    // --- System ---
    int spatial_dim = 1;    ///< 1, 2 or 3
    int n_electrons = 1;    ///< 1–4
    Representation representation = Representation::Exact;
    TrapKind trap = TrapKind::Harmonic;
    double omega = 1.0;     ///< Harmonic trap frequency
    double soft_a = 1.0;    ///< Soft-Coulomb regularisation length
    double Z = 1.0;         ///< Nuclear charge for TrapKind::SoftAtom
    double lambda_contact = 0.0; ///< Orbital-mode contact interaction λ ρ(r)
    bool fermion = false;   ///< Antisymmetrise exact two-electron 1D initial data

    // --- Initial wave function (displaced Gaussian, optional boost) ---
    double alpha = 1.0;     ///< Gaussian width parameter, ψ ~ exp(-α r² / 2)
    double x0 = 1.0;        ///< Displacement of electron 1 along x
    double k0 = 0.0;        ///< Initial momentum (boost) of electron 1 along x

    // --- Laser (dipole approximation) E(t) = E0 sin(ω_L t) × envelope ---
    double E0 = 0.0;
    double omega_L = 0.5;
    bool laser_envelope = false; ///< sin²(π t / T) envelope if true, CW otherwise

    // --- I/O ---
    std::string title;
    std::string prefix;
    std::string input_path;
    std::string output = "observables.csv";
    std::string plot_prefix; ///< If non-empty, write 1D line plots at t = 0 and t = T
    int n_plot_points = 800;
    int printlevel = 0;
    bool output_explicit = false; ///< true if output= was set (do not rewrite from prefix)
    bool smoke = false;     ///< Tiny run for ctest
    bool validate_free = false; ///< Compare 1D free particle with the analytic Gaussian
    bool validate_ho = false;   ///< Compare 1D HO coherent state (α = ω) with the analytic packet
    bool ident_check = true; ///< Apply IdentityConvolution once at t = 0 as a sanity check
    bool renormalize = false; ///< Project back onto the unit sphere after each step
    int nthreads = 0;       ///< OpenMP threads per MPI rank; 0 → OMP_NUM_THREADS / runtime default
};

Parameters parse_cli(int argc, char **argv);
void apply_smoke_defaults(Parameters &p);
int mra_dimension(const Parameters &p);
const char *propagator_name(Propagator p);
const char *kinetic_name(KineticKind k);
const char *representation_name(Representation r);
void print_parameters(const Parameters &p);

} // namespace tdse

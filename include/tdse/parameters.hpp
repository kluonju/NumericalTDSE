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

enum class JobKind {
    Tdse,   ///< Real-time i ∂t ψ = H ψ
    Ground, ///< Lowest eigenstate(s); default n_states = 1 (exact) or n_electrons (orbital)
    Eigen,  ///< Several lowest eigenstates; default n_states = 4 (exact) or n_electrons (orbital)
    Invert  ///< Interacting v_ext[n] inversion (TGK08 / Peirs), 2 electrons
};

enum class InvertTarget {
    Self, ///< Ground-state density of the namelist trap, then recover v_ext
    File  ///< Density from invert_density_file
};

enum class InvertGuess {
    Scaled,   ///< invert_scale × true trap (self-test)
    Harmonic, ///< ½ ω² r²
    Zero,     ///< v = 0
    Atom,     ///< −Z / sqrt(r² + a²)
    Hx        ///< v_s − ½ v_H (exact exchange, no correlation)
};

enum class EigenMethod {
    Lanczos, ///< Hermitian Ritz extraction from a Krylov space of H (linear; SCF outer loop if λρ ≠ 0)
    Itp      ///< Imaginary-time relaxation ψ(τ) ∝ exp(−H τ) ψ, Gram–Schmidt for excited states
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
    bool fermion = false;   ///< Antisymmetrise exact 2e/3e 1D initial data (Slater)
    bool ee = true;         ///< Exact N-body 1D: include soft-Coulomb V_ee

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

    // --- Stationary (ground / lowest eigenstates) ---
    JobKind job = JobKind::Tdse;
    EigenMethod eigen_method = EigenMethod::Lanczos;
    int n_states = 1;           ///< Number of lowest eigenstates to compute
    bool n_states_explicit = false;
    bool eigen_method_explicit = false;
    double eigen_thr = 1.0e-6;  ///< ITP / SCF: stop when |ΔE| is below this
    double eigen_residual = 0.0; ///< ||(H−E)ψ|| threshold; 0 → 50 × prec
    int eigen_maxiter = 80;     ///< ITP steps or SCF cycles

    // --- Density-to-potential inversion (TGK08 / Peirs, 2 electrons) ---
    InvertTarget invert_target = InvertTarget::Self;
    InvertGuess invert_guess = InvertGuess::Scaled;
    int n_grid = 0;                 ///< Points per spatial axis; 0 → 49 (1D) or 15 (2D)
    double invert_gamma = 0.25;     ///< Step γ in v ← v + γ (w0 + |r|^β) (n − n*)
    double invert_beta = 1.0;       ///< Tail weight exponent β (TGK08)
    double invert_w0 = 1.0;         ///< Weight floor; 0 → freeze v(0) as in TGK08
    double invert_tol = 1.0e-4;     ///< Stop when ∫ |n − n*| < this
    int invert_maxiter = 40;        ///< Outer inversion iterations
    int invert_inner = 40;          ///< Imag-time steps per outer iteration
    double invert_tau = 0.08;       ///< Imag-time step (implicit kinetic split)
    double invert_ncut = 1.0e-3;    ///< Density mask for v_s / v comparison
    double invert_scale = 0.55;     ///< Scaled-trap initial guess
    double invert_dvmax = 0.5;      ///< Clip |Δv| per outer step
    bool invert_check = false;      ///< Non-zero exit if the L1 error does not fall
    bool invert_ks_only = false;    ///< Skip interacting inversion; print v_s, v_H, v_c
    std::string invert_density_file; ///< Target density if invert_target = File
};

inline bool is_stationary(const Parameters &p) {
    return p.job == JobKind::Ground || p.job == JobKind::Eigen;
}

inline bool is_invert(const Parameters &p) { return p.job == JobKind::Invert; }

/** Physical lower bound used to discard spurious MW-kinetic Ritz values. */
inline double energy_floor(const Parameters &p) {
    if (p.trap == TrapKind::Harmonic) {
        return -0.05;
    }
    if (p.trap == TrapKind::SoftAtom) {
        return -0.5 * p.Z * p.Z - 10.0;
    }
    return -1.0;
}

Parameters parse_cli(int argc, char **argv);
void apply_smoke_defaults(Parameters &p);
int mra_dimension(const Parameters &p);
const char *propagator_name(Propagator p);
const char *kinetic_name(KineticKind k);
const char *representation_name(Representation r);
const char *job_kind_name(JobKind j);
const char *eigen_method_name(EigenMethod m);
const char *invert_target_name(InvertTarget t);
const char *invert_guess_name(InvertGuess g);
void print_parameters(const Parameters &p);

} // namespace tdse

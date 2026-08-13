#include "tdse/parameters.hpp"

#include "MRCPP/Printer"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace tdse {
namespace {

void usage(const char *argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options]\n"
        << "\nMRA / numerics\n"
        << "  --prec <double>          Adaptive MW precision (default 1e-4)\n"
        << "  --order <int>            Polynomial order k (default 7)\n"
        << "  --max-depth <int>        Maximum refinement depth (default 20)\n"
        << "  --L <double>             Half-box size, domain [-L, L] (default 8)\n"
        << "  --legendre               Use Legendre scaling functions (needed for split+TEO)\n"
        << "\nTime stepping\n"
        << "  --dt <double>            Time step (default 0.02)\n"
        << "  --T <double>             Final time (default 0.40)\n"
        << "  --propagator split|krylov|rk4   Default: krylov\n"
        << "  --kinetic abgv|bs|dconv  Derivative realisation of T (default abgv)\n"
        << "  --krylov-dim <int>       Lanczos subspace size (default 12)\n"
        << "  --teo-scale <int>        TimeEvolutionOperator finest scale (default 8)\n"
        << "  --renormalize            Restore ||ψ|| = 1 after each step\n"
        << "\nSystem\n"
        << "  --dim <1|2|3>            Spatial dimension (default 1)\n"
        << "  --electrons <1-4>        Number of electrons (default 1)\n"
        << "  --mode exact|orbital     exact: N-body tree (D=n_e*dim ≤ 3)\n"
        << "                           orbital: up to 4 orbitals (default exact)\n"
        << "  --trap harmonic|free|atom\n"
        << "  --omega <double>         Harmonic frequency (default 1)\n"
        << "  --soft-a <double>        Soft-Coulomb length (default 1)\n"
        << "  --Z <double>             Nuclear charge (default 1)\n"
        << "  --lambda <double>        Orbital contact interaction λ (default 0)\n"
        << "  --fermion                Antisymmetrise exact 2e-1D initial data\n"
        << "\nInitial state / laser\n"
        << "  --alpha <double>         Gaussian width (default 1)\n"
        << "  --x0 <double>            Displacement (default 1)\n"
        << "  --k0 <double>            Boost momentum (default 0)\n"
        << "  --E0 <double>            Laser amplitude (default 0)\n"
        << "  --omega-L <double>       Laser frequency (default 0.5)\n"
        << "  --envelope               sin²(π t/T) laser envelope\n"
        << "\nI/O\n"
        << "  --output <file>          CSV path (default observables.csv)\n"
        << "  --plot <prefix>          Write 1D line plots at t=0 and t=T\n"
        << "  --print-every <int>      Observable stride (default 1)\n"
        << "  --printlevel <int>       MRCPP printer verbosity (default 0)\n"
        << "  --no-ident-check         Skip IdentityConvolution diagnostic\n"
        << "  --validate-free          1D free Gaussian vs analytic solution\n"
        << "  --smoke                  Tiny RK4 run for ctest\n"
        << "  -h, --help               This message\n";
}

bool eq(const char *a, const char *b) { return std::strcmp(a, b) == 0; }

double require_double(int &i, int argc, char **argv) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + argv[i]);
    }
    return std::strtod(argv[++i], nullptr);
}

int require_int(int &i, int argc, char **argv) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + argv[i]);
    }
    return std::atoi(argv[++i]);
}

const char *require_str(int &i, int argc, char **argv) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + argv[i]);
    }
    return argv[++i];
}

} // namespace

void apply_smoke_defaults(Parameters &p) {
    p.prec = 1.0e-3;
    p.order = 5;
    p.max_depth = 12;
    p.L = 6.0;
    p.dt = 0.05;
    p.T = 0.10;
    p.propagator = Propagator::RK4;
    p.ident_check = false;
    p.spatial_dim = 1;
    p.n_electrons = 1;
    p.representation = Representation::Exact;
    p.trap = TrapKind::Harmonic;
    p.omega = 1.0;
    p.x0 = 0.5;
    p.alpha = 1.0;
    p.output = "smoke_observables.csv";
    p.print_every = 1;
    p.smoke = true;
}

int mra_dimension(const Parameters &p) {
    if (p.representation == Representation::Orbital) {
        return p.spatial_dim;
    }
    return p.spatial_dim * p.n_electrons;
}

const char *propagator_name(Propagator p) {
    switch (p) {
        case Propagator::Split:
            return "split";
        case Propagator::Krylov:
            return "krylov";
        case Propagator::RK4:
            return "rk4";
    }
    return "?";
}

const char *kinetic_name(KineticKind k) {
    switch (k) {
        case KineticKind::ABGV:
            return "abgv";
        case KineticKind::BS:
            return "bs";
        case KineticKind::DConv:
            return "dconv";
    }
    return "?";
}

const char *representation_name(Representation r) {
    return (r == Representation::Exact) ? "exact" : "orbital";
}

void print_parameters(const Parameters &p) {
    mrcpp::print::header(0, "NumericalTDSE parameters");
    mrcpp::print::value(0, "prec", p.prec);
    mrcpp::print::value(0, "order", static_cast<double>(p.order));
    mrcpp::print::value(0, "max_depth", static_cast<double>(p.max_depth));
    mrcpp::print::value(0, "L", p.L);
    mrcpp::print::value(0, "dt", p.dt);
    mrcpp::print::value(0, "T", p.T);
    mrcpp::print::value(0, "spatial_dim", static_cast<double>(p.spatial_dim));
    mrcpp::print::value(0, "n_electrons", static_cast<double>(p.n_electrons));
    mrcpp::print::value(0, "MRA dimension", static_cast<double>(mra_dimension(p)));
    println(0, "  propagator      : " << propagator_name(p.propagator));
    println(0, "  kinetic         : " << kinetic_name(p.kinetic));
    println(0, "  representation  : " << representation_name(p.representation));
    mrcpp::print::separator(0, '=', 2);
}

Parameters parse_cli(int argc, char **argv) {
    Parameters p;
    for (int i = 1; i < argc; ++i) {
        if (eq(argv[i], "-h") || eq(argv[i], "--help")) {
            usage(argv[0]);
            std::exit(0);
        } else if (eq(argv[i], "--prec")) {
            p.prec = require_double(i, argc, argv);
        } else if (eq(argv[i], "--order")) {
            p.order = require_int(i, argc, argv);
        } else if (eq(argv[i], "--max-depth")) {
            p.max_depth = require_int(i, argc, argv);
        } else if (eq(argv[i], "--L")) {
            p.L = require_double(i, argc, argv);
        } else if (eq(argv[i], "--legendre")) {
            p.use_legendre = true;
        } else if (eq(argv[i], "--dt")) {
            p.dt = require_double(i, argc, argv);
        } else if (eq(argv[i], "--T")) {
            p.T = require_double(i, argc, argv);
        } else if (eq(argv[i], "--propagator")) {
            const std::string s = require_str(i, argc, argv);
            if (s == "split") {
                p.propagator = Propagator::Split;
            } else if (s == "krylov") {
                p.propagator = Propagator::Krylov;
            } else if (s == "rk4") {
                p.propagator = Propagator::RK4;
            } else {
                throw std::invalid_argument("unknown propagator: " + s);
            }
        } else if (eq(argv[i], "--kinetic")) {
            const std::string s = require_str(i, argc, argv);
            if (s == "abgv") {
                p.kinetic = KineticKind::ABGV;
            } else if (s == "bs") {
                p.kinetic = KineticKind::BS;
            } else if (s == "dconv") {
                p.kinetic = KineticKind::DConv;
            } else {
                throw std::invalid_argument("unknown kinetic: " + s);
            }
        } else if (eq(argv[i], "--krylov-dim")) {
            p.krylov_dim = require_int(i, argc, argv);
        } else if (eq(argv[i], "--teo-scale")) {
            p.teo_finest_scale = require_int(i, argc, argv);
        } else if (eq(argv[i], "--renormalize")) {
            p.renormalize = true;
        } else if (eq(argv[i], "--dim")) {
            p.spatial_dim = require_int(i, argc, argv);
        } else if (eq(argv[i], "--electrons")) {
            p.n_electrons = require_int(i, argc, argv);
        } else if (eq(argv[i], "--mode")) {
            const std::string s = require_str(i, argc, argv);
            if (s == "exact") {
                p.representation = Representation::Exact;
            } else if (s == "orbital") {
                p.representation = Representation::Orbital;
            } else {
                throw std::invalid_argument("unknown mode: " + s);
            }
        } else if (eq(argv[i], "--trap")) {
            const std::string s = require_str(i, argc, argv);
            if (s == "harmonic") {
                p.trap = TrapKind::Harmonic;
            } else if (s == "free") {
                p.trap = TrapKind::None;
            } else if (s == "atom") {
                p.trap = TrapKind::SoftAtom;
            } else {
                throw std::invalid_argument("unknown trap: " + s);
            }
        } else if (eq(argv[i], "--omega")) {
            p.omega = require_double(i, argc, argv);
        } else if (eq(argv[i], "--soft-a")) {
            p.soft_a = require_double(i, argc, argv);
        } else if (eq(argv[i], "--Z")) {
            p.Z = require_double(i, argc, argv);
        } else if (eq(argv[i], "--lambda")) {
            p.lambda_contact = require_double(i, argc, argv);
        } else if (eq(argv[i], "--fermion")) {
            p.fermion = true;
        } else if (eq(argv[i], "--alpha")) {
            p.alpha = require_double(i, argc, argv);
        } else if (eq(argv[i], "--x0")) {
            p.x0 = require_double(i, argc, argv);
        } else if (eq(argv[i], "--k0")) {
            p.k0 = require_double(i, argc, argv);
        } else if (eq(argv[i], "--E0")) {
            p.E0 = require_double(i, argc, argv);
        } else if (eq(argv[i], "--omega-L")) {
            p.omega_L = require_double(i, argc, argv);
        } else if (eq(argv[i], "--envelope")) {
            p.laser_envelope = true;
        } else if (eq(argv[i], "--output")) {
            p.output = require_str(i, argc, argv);
        } else if (eq(argv[i], "--plot")) {
            p.plot_prefix = require_str(i, argc, argv);
        } else if (eq(argv[i], "--print-every")) {
            p.print_every = require_int(i, argc, argv);
        } else if (eq(argv[i], "--printlevel")) {
            p.printlevel = require_int(i, argc, argv);
        } else if (eq(argv[i], "--no-ident-check")) {
            p.ident_check = false;
        } else if (eq(argv[i], "--validate-free")) {
            p.validate_free = true;
            p.trap = TrapKind::None;
            p.E0 = 0.0;
        } else if (eq(argv[i], "--smoke")) {
            apply_smoke_defaults(p);
        } else {
            throw std::invalid_argument(std::string("unknown option: ") + argv[i]);
        }
    }

    if (p.spatial_dim < 1 || p.spatial_dim > 3) {
        throw std::invalid_argument("--dim must be 1, 2 or 3");
    }
    if (p.n_electrons < 1 || p.n_electrons > 4) {
        throw std::invalid_argument("--electrons must be 1..4");
    }
    if (p.representation == Representation::Exact) {
        const int D = mra_dimension(p);
        if (D > 3) {
            throw std::invalid_argument(
                    "exact N-body requires n_electrons * dim <= 3; use --mode orbital for 4 electrons");
        }
    }
    if (p.propagator == Propagator::Split && mra_dimension(p) == 1 && !p.use_legendre) {
        // TimeEvolutionOperator is implemented for Legendre scaling functions.
        p.use_legendre = true;
    }
    if (p.dt <= 0.0 || p.T < 0.0) {
        throw std::invalid_argument("dt must be positive and T >= 0");
    }
    return p;
}

} // namespace tdse

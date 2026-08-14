#include "tdse/input.hpp"
#include "tdse/parameters.hpp"
#include "tdse/parallel.hpp"

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
        << "Usage: " << argv0 << " <input.in>\n"
        << "       " << argv0 << " -i <input.in>\n"
        << "       " << argv0 << " --template\n"
        << "       " << argv0 << " --smoke\n"
        << "\n"
        << "  Input files use Quantum ESPRESSO namelists:  &SECTION ... /\n"
        << "  Only the keywords you need; everything else keeps its default.\n"
        << "\n"
        << "  -i, --input <file>   Namelist input (same as a positional filename)\n"
        << "  --template           Print a commented template to stdout\n"
        << "  --smoke              Tiny built-in RK4 run (ctest); no input file\n"
        << "  -h, --help           This message\n"
        << "\n"
        << "Namelists: &CONTROL &MRA &TIME &SYSTEM &INITIAL &LASER &OUTPUT &PARALLEL\n"
        << "Hybrid MPI+OpenMP: mpirun -np <ranks> --bind-to core:overload-allowed \\\n"
        << "                   -x OMP_NUM_THREADS=<threads> " << argv0 << " job.in\n"
        << "Run `" << argv0 << " --template` for keywords and defaults.\n";
}

bool eq(const char *a, const char *b) { return std::strcmp(a, b) == 0; }

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
    if (!p.title.empty()) {
        println(0, "  title           : " << p.title);
    }
    if (!p.input_path.empty()) {
        println(0, "  input           : " << p.input_path);
    }
    mrcpp::print::value(0, "prec", p.prec);
    mrcpp::print::value(0, "order", static_cast<double>(p.order));
    mrcpp::print::value(0, "max_depth", static_cast<double>(p.max_depth));
    mrcpp::print::value(0, "L", p.L);
    mrcpp::print::value(0, "dt", p.dt);
    mrcpp::print::value(0, "T", p.T);
    mrcpp::print::value(0, "spatial_dim", static_cast<double>(p.spatial_dim));
    mrcpp::print::value(0, "n_electrons", static_cast<double>(p.n_electrons));
    mrcpp::print::value(0, "MRA dimension", static_cast<double>(mra_dimension(p)));
    println(0, "  basis           : " << basis_name(p));
    println(0, "  propagator      : " << propagator_name(p.propagator));
    println(0, "  kinetic         : " << kinetic_name(p.kinetic));
    println(0, "  representation  : " << representation_name(p.representation));
    println(0, "  trap            : " << trap_name(p.trap));
    println(0, "  output          : " << p.output);
    mrcpp::print::value(0, "MPI ranks", static_cast<double>(parallel::size));
    mrcpp::print::value(0, "OpenMP threads / rank", static_cast<double>(parallel::nthreads));
    mrcpp::print::separator(0, '=', 2);
}

Parameters parse_cli(int argc, char **argv) {
    Parameters p;
    std::string input_file;
    bool smoke_cli = false;

    for (int i = 1; i < argc; ++i) {
        if (eq(argv[i], "-h") || eq(argv[i], "--help")) {
            if (parallel::io_rank()) {
                usage(argv[0]);
            }
            parallel::shutdown(0);
        } else if (eq(argv[i], "--template")) {
            if (parallel::io_rank()) {
                write_input_template(std::cout);
            }
            parallel::shutdown(0);
        } else if (eq(argv[i], "--smoke")) {
            smoke_cli = true;
        } else if (eq(argv[i], "-i") || eq(argv[i], "--input")) {
            input_file = require_str(i, argc, argv);
        } else if (argv[i][0] != '-') {
            if (!input_file.empty()) {
                throw std::invalid_argument("multiple input files: '" + input_file + "' and '" + argv[i] + "'");
            }
            input_file = argv[i];
        } else {
            throw std::invalid_argument(std::string("unknown option: ") + argv[i] + " (input is a namelist file; see --help)");
        }
    }

    if (smoke_cli) {
        apply_smoke_defaults(p);
    } else if (input_file.empty()) {
        if (parallel::io_rank()) {
            usage(argv[0]);
        }
        parallel::shutdown(1);
    }

    if (!input_file.empty()) {
        parse_namelist_file(input_file, p);
        if (p.smoke) {
            // calculation='smoke' / smoke=.true. loads the tiny-run preset, then
            // the file is read again so any explicit keys still win.
            apply_smoke_defaults(p);
            parse_namelist_file(input_file, p);
        }
    }

    finalize_parameters(p);
    return p;
}

} // namespace tdse

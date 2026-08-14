#include "tdse/input.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tdse {
namespace {

std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string to_upper_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

std::string canonical_section(std::string name) {
    name = to_upper_copy(std::move(name));
    if (name == "CTRL") {
        return "CONTROL";
    }
    if (name == "GRID" || name == "NUMERICS") {
        return "MRA";
    }
    if (name == "PROPAGATOR") {
        return "TIME";
    }
    if (name == "WAVEFUNCTION" || name == "PSI") {
        return "INITIAL";
    }
    if (name == "FIELD") {
        return "LASER";
    }
    if (name == "IO" || name == "OUT") {
        return "OUTPUT";
    }
    if (name == "PARA" || name == "OMP") {
        return "PARALLEL";
    }
    if (name == "GROUND" || name == "GS" || name == "EIGENSTATE" || name == "EIGENSTATES" || name == "SCF") {
        return "EIGEN";
    }
    return name;
}

const std::unordered_set<std::string> kKnownSections = {
        "CONTROL", "MRA", "TIME", "SYSTEM", "INITIAL", "LASER", "OUTPUT", "PARALLEL", "EIGEN"};

bool is_ident_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

void skip_ws(const std::string &s, std::size_t &i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
        ++i;
    }
}

void skip_ws_comma(const std::string &s, std::size_t &i) {
    while (i < s.size() && (std::isspace(static_cast<unsigned char>(s[i])) || s[i] == ',')) {
        ++i;
    }
}

std::string strip_comment(const std::string &line) {
    std::string out;
    out.reserve(line.size());
    char quote = '\0';
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quote) {
            out.push_back(c);
            if (c == quote) {
                if (quote == '\'' && i + 1 < line.size() && line[i + 1] == '\'') {
                    out.push_back(line[++i]);
                    continue;
                }
                quote = '\0';
            }
            continue;
        }
        if (c == '\'' || c == '"') {
            quote = c;
            out.push_back(c);
            continue;
        }
        if (c == '!' || c == '#') {
            break;
        }
        out.push_back(c);
    }
    return out;
}

std::string read_ident(const std::string &s, std::size_t &i, const std::string &origin) {
    if (i >= s.size() || !(std::isalpha(static_cast<unsigned char>(s[i])) || s[i] == '_')) {
        throw std::invalid_argument(origin + ": expected identifier");
    }
    const std::size_t start = i;
    ++i;
    while (i < s.size() && is_ident_char(s[i])) {
        ++i;
    }
    return s.substr(start, i - start);
}

std::string read_value(const std::string &s, std::size_t &i, const std::string &origin) {
    if (i >= s.size()) {
        throw std::invalid_argument(origin + ": expected value after '='");
    }
    if (s[i] == '\'' || s[i] == '"') {
        const char q = s[i++];
        std::string val;
        while (i < s.size()) {
            if (s[i] == q) {
                if (q == '\'' && i + 1 < s.size() && s[i + 1] == '\'') {
                    val.push_back('\'');
                    i += 2;
                    continue;
                }
                ++i;
                return val;
            }
            val.push_back(s[i++]);
        }
        throw std::invalid_argument(origin + ": unterminated string");
    }
    const std::size_t start = i;
    while (i < s.size()) {
        const char c = s[i];
        if (c == ',' || c == '/' || c == '&' || std::isspace(static_cast<unsigned char>(c))) {
            break;
        }
        ++i;
    }
    if (i == start) {
        throw std::invalid_argument(origin + ": empty value");
    }
    return s.substr(start, i - start);
}

std::string fortran_number(std::string v) {
    for (std::size_t i = 1; i + 1 < v.size(); ++i) {
        if ((v[i] == 'd' || v[i] == 'D') && (std::isdigit(static_cast<unsigned char>(v[i - 1])) || v[i - 1] == '.') &&
            (std::isdigit(static_cast<unsigned char>(v[i + 1])) || v[i + 1] == '+' || v[i + 1] == '-')) {
            v[i] = (v[i] == 'd') ? 'e' : 'E';
        }
    }
    return v;
}

bool parse_bool(const std::string &raw, const std::string &origin) {
    const std::string v = to_lower_copy(raw);
    if (v == ".true." || v == ".t." || v == "true" || v == "t" || v == "yes" || v == "on" || v == "1") {
        return true;
    }
    if (v == ".false." || v == ".f." || v == "false" || v == "f" || v == "no" || v == "off" || v == "0") {
        return false;
    }
    throw std::invalid_argument(origin + ": not a boolean: " + raw);
}

double parse_double(const std::string &raw, const std::string &origin) {
    const std::string v = fortran_number(raw);
    char *end = nullptr;
    const double x = std::strtod(v.c_str(), &end);
    if (end == v.c_str() || *end != '\0') {
        throw std::invalid_argument(origin + ": not a real number: " + raw);
    }
    return x;
}

int parse_int(const std::string &raw, const std::string &origin) {
    const std::string v = fortran_number(raw);
    char *end = nullptr;
    const long x = std::strtol(v.c_str(), &end, 10);
    if (end == v.c_str() || *end != '\0') {
        throw std::invalid_argument(origin + ": not an integer: " + raw);
    }
    return static_cast<int>(x);
}

std::string unquote(const std::string &raw) { return raw; }

void set_propagator(Parameters &p, const std::string &raw, const std::string &origin) {
    const std::string s = to_lower_copy(raw);
    if (s == "split" || s == "strang") {
        p.propagator = Propagator::Split;
    } else if (s == "krylov" || s == "sil" || s == "lanczos") {
        p.propagator = Propagator::Krylov;
    } else if (s == "rk4" || s == "runge-kutta" || s == "runge_kutta") {
        p.propagator = Propagator::RK4;
    } else {
        throw std::invalid_argument(origin + ": unknown propagator '" + raw + "' (split|krylov|rk4)");
    }
}

void set_kinetic(Parameters &p, const std::string &raw, const std::string &origin) {
    const std::string s = to_lower_copy(raw);
    if (s == "abgv") {
        p.kinetic = KineticKind::ABGV;
    } else if (s == "bs" || s == "bspline") {
        p.kinetic = KineticKind::BS;
    } else if (s == "dconv" || s == "derivative_convolution" || s == "conv") {
        p.kinetic = KineticKind::DConv;
    } else {
        throw std::invalid_argument(origin + ": unknown kinetic '" + raw + "' (abgv|bs|dconv)");
    }
}

void set_mode(Parameters &p, const std::string &raw, const std::string &origin) {
    const std::string s = to_lower_copy(raw);
    if (s == "exact" || s == "nbody" || s == "full") {
        p.representation = Representation::Exact;
    } else if (s == "orbital" || s == "orbitals" || s == "meanfield" || s == "hartree") {
        p.representation = Representation::Orbital;
    } else {
        throw std::invalid_argument(origin + ": unknown mode '" + raw + "' (exact|orbital)");
    }
}

void set_trap(Parameters &p, const std::string &raw, const std::string &origin) {
    const std::string s = to_lower_copy(raw);
    if (s == "harmonic" || s == "ho" || s == "oscillator") {
        p.trap = TrapKind::Harmonic;
    } else if (s == "free" || s == "none" || s == "particle") {
        p.trap = TrapKind::None;
    } else if (s == "atom" || s == "soft" || s == "coulomb" || s == "softatom") {
        p.trap = TrapKind::SoftAtom;
    } else {
        throw std::invalid_argument(origin + ": unknown trap '" + raw + "' (harmonic|free|atom)");
    }
}

void set_basis(Parameters &p, const std::string &raw, const std::string &origin) {
    const std::string s = to_lower_copy(raw);
    if (s == "interpolating" || s == "interp" || s == "mw") {
        p.use_legendre = false;
    } else if (s == "legendre" || s == "legen") {
        p.use_legendre = true;
    } else {
        throw std::invalid_argument(origin + ": unknown basis '" + raw + "' (interpolating|legendre)");
    }
}

void set_eigen_method(Parameters &p, const std::string &raw, const std::string &origin) {
    const std::string s = to_lower_copy(raw);
    if (s == "lanczos" || s == "ritz" || s == "krylov") {
        p.eigen_method = EigenMethod::Lanczos;
        p.eigen_method_explicit = true;
    } else if (s == "itp" || s == "imag" || s == "imaginary" || s == "relaxation" || s == "filter") {
        p.eigen_method = EigenMethod::Itp;
        p.eigen_method_explicit = true;
    } else {
        throw std::invalid_argument(origin + ": unknown eigen method '" + raw + "' (lanczos|itp)");
    }
}

void set_calculation(Parameters &p, const std::string &raw, const std::string &origin) {
    const std::string s = to_lower_copy(raw);
    if (s == "tdse") {
        p.job = JobKind::Tdse;
        p.smoke = false;
    } else if (s == "smoke") {
        p.job = JobKind::Tdse;
        p.smoke = true;
    } else if (s == "ground" || s == "gs" || s == "scf") {
        p.job = JobKind::Ground;
        p.smoke = false;
    } else if (s == "eigen" || s == "eigenstate" || s == "eigenstates" || s == "stationary") {
        p.job = JobKind::Eigen;
        p.smoke = false;
    } else {
        throw std::invalid_argument(origin + ": calculation must be 'tdse', 'ground', 'eigen' or 'smoke'");
    }
}

bool is_control_like(const std::string &section) { return section == "CONTROL" || section == "OUTPUT"; }

bool apply_eigen_keyword(Parameters &p, const std::string &key, const std::string &value, const std::string &origin) {
    if (key == "n_states" || key == "nstates" || key == "n_eigen" || key == "states") {
        p.n_states = parse_int(value, origin);
        p.n_states_explicit = true;
        return true;
    }
    if (key == "eigen_method" || key == "method" || key == "eigensolver") {
        set_eigen_method(p, value, origin);
        return true;
    }
    if (key == "conv_thr" || key == "thr" || key == "eigen_thr") {
        p.eigen_thr = parse_double(value, origin);
        return true;
    }
    if (key == "residual" || key == "residual_thr" || key == "res_thr") {
        p.eigen_residual = parse_double(value, origin);
        return true;
    }
    if (key == "max_iter" || key == "maxiter" || key == "scf_iter") {
        p.eigen_maxiter = parse_int(value, origin);
        return true;
    }
    return false;
}

} // namespace

void apply_namelist_assignment(Parameters &p,
                               const std::string &section_in,
                               const std::string &key_in,
                               const std::string &value,
                               const std::string &origin) {
    const std::string section = canonical_section(section_in);
    const std::string key = to_lower_copy(key_in);

    auto unknown = [&]() {
        throw std::invalid_argument(origin + ": unknown keyword '" + key + "' in &" + section);
    };

    if (is_control_like(section)) {
        if (key == "calculation") {
            set_calculation(p, value, origin);
        } else if (key == "title") {
            p.title = unquote(value);
        } else if (key == "prefix") {
            p.prefix = unquote(value);
        } else if (key == "output" || key == "outfile" || key == "observables") {
            p.output = unquote(value);
            p.output_explicit = true;
        } else if (key == "plot" || key == "plot_prefix") {
            p.plot_prefix = unquote(value);
        } else if (key == "n_plot" || key == "n_plot_points" || key == "nplot") {
            p.n_plot_points = parse_int(value, origin);
        } else if (key == "printlevel" || key == "verbosity" || key == "verbose") {
            p.printlevel = parse_int(value, origin);
        } else if (key == "print_every" || key == "iprint") {
            p.print_every = parse_int(value, origin);
        } else if (key == "ident_check" || key == "identcheck") {
            p.ident_check = parse_bool(value, origin);
        } else if (key == "renormalize") {
            p.renormalize = parse_bool(value, origin);
        } else if (key == "validate_free" || key == "validatefree") {
            p.validate_free = parse_bool(value, origin);
        } else if (key == "validate_ho" || key == "validateho") {
            p.validate_ho = parse_bool(value, origin);
        } else if (key == "validate") {
            const std::string s = to_lower_copy(value);
            if (s == "free") {
                p.validate_free = true;
            } else if (s == "ho" || s == "harmonic") {
                p.validate_ho = true;
            } else if (s == "none" || s == "off" || s.empty()) {
                p.validate_free = false;
                p.validate_ho = false;
            } else {
                throw std::invalid_argument(origin + ": validate must be 'free', 'ho' or 'none'");
            }
        } else if (key == "smoke") {
            p.smoke = parse_bool(value, origin);
        } else if (key == "nthreads" || key == "omp_threads" || key == "threads") {
            p.nthreads = parse_int(value, origin);
        } else if (apply_eigen_keyword(p, key, value, origin)) {
            return;
        } else {
            unknown();
        }
        return;
    }

    if (section == "PARALLEL") {
        if (key == "nthreads" || key == "omp_threads" || key == "threads" || key == "omp") {
            p.nthreads = parse_int(value, origin);
        } else {
            unknown();
        }
        return;
    }

    if (section == "EIGEN") {
        if (!apply_eigen_keyword(p, key, value, origin)) {
            unknown();
        }
        return;
    }

    if (section == "MRA") {
        if (key == "prec" || key == "precision") {
            p.prec = parse_double(value, origin);
        } else if (key == "order" || key == "k") {
            p.order = parse_int(value, origin);
        } else if (key == "max_depth" || key == "maxdepth" || key == "depth") {
            p.max_depth = parse_int(value, origin);
        } else if (key == "l" || key == "box" || key == "half_box") {
            p.L = parse_double(value, origin);
        } else if (key == "basis") {
            set_basis(p, value, origin);
        } else if (key == "legendre") {
            p.use_legendre = parse_bool(value, origin);
        } else {
            unknown();
        }
        return;
    }

    if (section == "TIME") {
        if (key == "dt") {
            p.dt = parse_double(value, origin);
        } else if (key == "t" || key == "t_max" || key == "tmax" || key == "tfinal") {
            p.T = parse_double(value, origin);
        } else if (key == "propagator" || key == "propagator_type") {
            set_propagator(p, value, origin);
        } else if (key == "kinetic") {
            set_kinetic(p, value, origin);
        } else if (key == "krylov_dim" || key == "krylovdim") {
            p.krylov_dim = parse_int(value, origin);
        } else if (key == "teo_scale" || key == "teo_finest_scale") {
            p.teo_finest_scale = parse_int(value, origin);
        } else if (key == "teo_jpower" || key == "jpower") {
            p.teo_jpower = parse_int(value, origin);
        } else if (key == "print_every" || key == "iprint") {
            p.print_every = parse_int(value, origin);
        } else {
            unknown();
        }
        return;
    }

    if (section == "SYSTEM") {
        if (key == "dim" || key == "spatial_dim" || key == "ndim") {
            p.spatial_dim = parse_int(value, origin);
        } else if (key == "electrons" || key == "n_electrons" || key == "nelec") {
            p.n_electrons = parse_int(value, origin);
        } else if (key == "mode" || key == "representation") {
            set_mode(p, value, origin);
        } else if (key == "trap" || key == "potential") {
            set_trap(p, value, origin);
        } else if (key == "omega") {
            p.omega = parse_double(value, origin);
        } else if (key == "soft_a" || key == "a_soft" || key == "asoft") {
            p.soft_a = parse_double(value, origin);
        } else if (key == "z") {
            p.Z = parse_double(value, origin);
        } else if (key == "lambda" || key == "lambda_contact") {
            p.lambda_contact = parse_double(value, origin);
        } else if (key == "fermion") {
            p.fermion = parse_bool(value, origin);
        } else if (key == "ee" || key == "interact" || key == "interaction" || key == "vee") {
            p.ee = parse_bool(value, origin);
        } else {
            unknown();
        }
        return;
    }

    if (section == "INITIAL") {
        if (key == "alpha") {
            p.alpha = parse_double(value, origin);
        } else if (key == "x0") {
            p.x0 = parse_double(value, origin);
        } else if (key == "k0") {
            p.k0 = parse_double(value, origin);
        } else {
            unknown();
        }
        return;
    }

    if (section == "LASER") {
        if (key == "e0" || key == "amplitude") {
            p.E0 = parse_double(value, origin);
        } else if (key == "omega_l" || key == "omegal" || key == "wl") {
            p.omega_L = parse_double(value, origin);
        } else if (key == "envelope") {
            p.laser_envelope = parse_bool(value, origin);
        } else {
            unknown();
        }
        return;
    }

    throw std::invalid_argument(origin + ": unknown namelist &" + section_in);
}

void parse_namelist_file(const std::string &path, Parameters &p) {
    std::ifstream in(path);
    if (!in) {
        throw std::invalid_argument("cannot open input file: " + path);
    }
    p.input_path = path;

    std::string section;
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::string origin = path + ":" + std::to_string(line_no);
        const std::string text = strip_comment(line);
        std::size_t i = 0;
        skip_ws(text, i);
        if (i >= text.size()) {
            continue;
        }

        while (i < text.size()) {
            skip_ws_comma(text, i);
            if (i >= text.size()) {
                break;
            }
            if (text[i] == '/') {
                if (section.empty()) {
                    throw std::invalid_argument(origin + ": '/' outside a namelist");
                }
                section.clear();
                ++i;
                continue;
            }
            if (text[i] == '&') {
                if (!section.empty()) {
                    throw std::invalid_argument(origin + ": namelist &" + section + " was not closed with '/'");
                }
                ++i;
                skip_ws(text, i);
                section = canonical_section(read_ident(text, i, origin));
                if (!kKnownSections.count(section)) {
                    throw std::invalid_argument(origin + ": unknown namelist &" + section);
                }
                continue;
            }
            if (section.empty()) {
                throw std::invalid_argument(origin + ": assignment outside a namelist (expected &SECTION ... /)");
            }
            const std::string key = read_ident(text, i, origin);
            skip_ws(text, i);
            if (i >= text.size() || text[i] != '=') {
                throw std::invalid_argument(origin + ": expected '=' after keyword '" + key + "'");
            }
            ++i;
            skip_ws(text, i);
            const std::string val = read_value(text, i, origin);
            apply_namelist_assignment(p, section, key, val, origin);
        }
    }
    if (!section.empty()) {
        throw std::invalid_argument(path + ": namelist &" + section + " was not closed with '/'");
    }
}

void finalize_parameters(Parameters &p) {
    if (p.validate_free && p.validate_ho) {
        throw std::invalid_argument("choose only one of validate_free and validate_ho");
    }
    if (p.validate_free) {
        p.trap = TrapKind::None;
        p.E0 = 0.0;
    }
    if (p.validate_ho) {
        p.trap = TrapKind::Harmonic;
        if (p.omega <= 0.0) {
            throw std::invalid_argument("validate_ho requires SYSTEM omega > 0");
        }
    }
    if (!p.n_states_explicit && is_stationary(p)) {
        if (p.representation == Representation::Orbital) {
            p.n_states = p.n_electrons;
        } else if (p.job == JobKind::Eigen) {
            p.n_states = 4;
        } else {
            p.n_states = 1;
        }
    }
    if (is_stationary(p)) {
        if (p.E0 != 0.0) {
            throw std::invalid_argument("ground/eigen calculations require LASER E0 = 0");
        }
        if (p.n_states < 1 || p.n_states > 12) {
            throw std::invalid_argument("EIGEN n_states must be 1..12");
        }
        if (p.eigen_maxiter < 1) {
            throw std::invalid_argument("EIGEN max_iter must be >= 1");
        }
        if (p.eigen_method == EigenMethod::Lanczos) {
            const int need = std::max(std::max(p.n_states + 6, 2 * p.n_states), 8);
            if (p.krylov_dim < need) {
                p.krylov_dim = need;
            }
        }
    }
    if (!p.prefix.empty() && !p.output_explicit) {
        p.output = p.prefix + "_observables.csv";
    }
    if (p.spatial_dim < 1 || p.spatial_dim > 3) {
        throw std::invalid_argument("SYSTEM dim must be 1, 2 or 3");
    }
    if (p.n_electrons < 1 || p.n_electrons > 4) {
        throw std::invalid_argument("SYSTEM electrons must be 1..4");
    }
    if (p.representation == Representation::Exact) {
        const int D = mra_dimension(p);
        if (D > 3) {
            throw std::invalid_argument(
                    "exact N-body requires electrons * dim <= 3; set mode = 'orbital' for 4 electrons");
        }
    }
    if (p.propagator == Propagator::Split && mra_dimension(p) == 1 && !p.use_legendre) {
        p.use_legendre = true;
    }
    if (p.dt <= 0.0 || p.T < 0.0) {
        throw std::invalid_argument("TIME dt must be positive and T >= 0");
    }
    if (p.order < 2) {
        throw std::invalid_argument("MRA order must be >= 2");
    }
    if (p.nthreads < 0) {
        throw std::invalid_argument("PARALLEL nthreads must be >= 0 (0 = OMP_NUM_THREADS)");
    }
}

void write_input_template(std::ostream &os) {
    os <<
            R"INP(! NumericalTDSE input  (Quantum ESPRESSO namelist style)
! Only the keywords you need; omitted keys keep their defaults.
! Comments start with ! or #.  Reals accept Fortran 1.0d-4.
! Booleans: .true. / .false.   Strings: 'quoted' or unquoted.

&CONTROL
  calculation   = 'tdse'          ! 'tdse' | 'ground' | 'eigen' | 'smoke'
  title         = 'job'
  prefix        = 'job'           ! default output = prefix_observables.csv
  printlevel    = 0
  print_every   = 1
  ident_check   = .true.
  renormalize   = .false.
  validate_free = .false.         ! overlap vs analytic free Gaussian
  validate_ho   = .false.         ! TDSE: coherent state; ground/eigen: HO ψ_n
  n_states      = 1               ! lowest eigenstates (ground/eigen jobs)
/

&MRA
  prec      = 1.0d-4              ! adaptive wavelet-norm threshold
  order     = 7                   ! polynomial order k
  max_depth = 20
  L         = 8.0                 ! domain [-L, L]
  basis     = 'interpolating'     ! 'interpolating' | 'legendre'
/

&TIME
  dt         = 0.02
  T          = 0.40
  propagator = 'krylov'           ! 'krylov' | 'split' | 'rk4'
  kinetic    = 'abgv'             ! 'abgv' | 'bs' | 'dconv'
  krylov_dim = 12
  teo_scale  = 8                  ! TimeEvolutionOperator finest scale
/

&SYSTEM
  dim       = 1                   ! spatial dimension 1|2|3
  electrons = 1                   ! 1..4
  mode      = 'exact'             ! 'exact' (N-body tree) | 'orbital'
  trap      = 'harmonic'          ! 'harmonic' | 'free' | 'atom'
  omega     = 1.0
  soft_a    = 1.0
  Z         = 1.0
  lambda    = 0.0                 ! orbital-mode contact interaction
  fermion   = .false.             ! Slater initial data for exact 2e/3e in 1D
  ee        = .true.              ! exact N-body 1D soft-Coulomb V_ee
/

&INITIAL
  alpha = 1.0                     ! Gaussian width, ψ ~ exp(-α r² / 2)
  x0    = 1.0                     ! displacement along x
  k0    = 0.0                     ! boost momentum
/

&LASER
  E0       = 0.0                  ! dipole field amplitude
  omega_L  = 0.5
  envelope = .false.              ! .true. → sin²(π t / T)
/

&OUTPUT
  output = 'observables.csv'
  plot   = ''                     ! non-empty → 1D line plots at t=0 and t=T
  n_plot = 800
/

&PARALLEL
  nthreads = 0                    ! OpenMP threads per MPI rank; 0 → OMP_NUM_THREADS
/

&EIGEN
  n_states   = 1                  ! ignored unless calculation = 'ground' or 'eigen'
  method     = 'lanczos'          ! 'lanczos' (Ritz) | 'itp' (imaginary time)
  conv_thr   = 1.0d-6             ! ITP/SCF |ΔE| threshold
  residual   = 0.0                ! ||(H-E)ψ||; 0 → 50*prec
  max_iter   = 80
/
)INP";
}

const char *trap_name(TrapKind t) {
    switch (t) {
        case TrapKind::Harmonic:
            return "harmonic";
        case TrapKind::None:
            return "free";
        case TrapKind::SoftAtom:
            return "atom";
    }
    return "?";
}

const char *basis_name(const Parameters &p) { return p.use_legendre ? "legendre" : "interpolating"; }

} // namespace tdse

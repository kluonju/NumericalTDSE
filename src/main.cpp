/**
 * NumericalTDSE — adaptive multiwavelet solver of the time-dependent
 * Schrödinger equation, built on MRCPP.
 *
 * Default demo: 1D harmonic oscillator, displaced Gaussian, Krylov propagator.
 * See README.md and `tdse --help`.
 */

#include "tdse/parameters.hpp"
#include "tdse/simulate.hpp"

#include "MRCPP/Printer"

#include <exception>
#include <iostream>

int main(int argc, char **argv) {
    try {
        tdse::Parameters p = tdse::parse_cli(argc, argv);
        mrcpp::Printer::init(p.printlevel);
        mrcpp::print::environment(0);
        return tdse::run(p);
    } catch (const std::exception &ex) {
        std::cerr << "NumericalTDSE error: " << ex.what() << std::endl;
        return 1;
    }
}

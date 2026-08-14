/**
 * NumericalTDSE — adaptive multiwavelet solver of the time-dependent
 * Schrödinger equation, built on MRCPP.
 *
 * Default demo: 1D harmonic oscillator, displaced Gaussian, Krylov propagator.
 * Input is a Quantum ESPRESSO-style namelist file; see README.md and `tdse --help`.
 */

#include "tdse/parameters.hpp"
#include "tdse/simulate.hpp"
#include "tdse/parallel.hpp"

#include "MRCPP/Printer"

#include <exception>
#include <iostream>

int main(int argc, char **argv) {
    tdse::parallel::init(argc, argv);
    try {
        tdse::Parameters p = tdse::parse_cli(argc, argv);
        tdse::parallel::configure_threads(p.nthreads);
        mrcpp::Printer::init(p.printlevel, tdse::parallel::rank, tdse::parallel::size);
        mrcpp::print::environment(0);
        const int rc = tdse::run(p);
        tdse::parallel::finalize();
        return rc;
    } catch (const std::exception &ex) {
        if (tdse::parallel::io_rank()) {
            std::cerr << "NumericalTDSE error: " << ex.what() << std::endl;
        }
        tdse::parallel::abort_all(1);
    }
}

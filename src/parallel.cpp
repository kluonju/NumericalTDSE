#include "tdse/parallel.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Core>

namespace tdse {
namespace parallel {
namespace {

bool we_initialized_mpi = false;

int env_max_threads() {
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

} // namespace

void init(int &argc, char **argv) {
    Eigen::setNbThreads(1);
#ifdef _OPENMP
    omp_set_dynamic(0);
#endif

#ifdef MRCPP_HAS_MPI
    int already = 0;
    MPI_Initialized(&already);
    if (!already) {
        int provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        we_initialized_mpi = true;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (provided < MPI_THREAD_FUNNELED && rank == 0) {
            std::cerr << "NumericalTDSE warning: MPI_THREAD_FUNNELED not provided; "
                         "hybrid OpenMP+MPI may be unsafe\n";
        }
    } else {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
#else
    (void)argc;
    (void)argv;
    rank = 0;
    size = 1;
#endif
    nthreads = env_max_threads();
    mrcpp::set_max_threads(nthreads);
}

void configure_threads(int nthreads_requested) {
    if (nthreads_requested > 0) {
        nthreads = nthreads_requested;
    } else {
        nthreads = env_max_threads();
    }
    nthreads = std::max(1, nthreads);
    mrcpp::set_max_threads(nthreads);
}

void finalize() {
#ifdef MRCPP_HAS_MPI
    int fini = 0;
    MPI_Finalized(&fini);
    if (!fini) {
        int ini = 0;
        MPI_Initialized(&ini);
        if (ini && we_initialized_mpi) {
            MPI_Finalize();
            we_initialized_mpi = false;
        }
    }
#endif
}

void shutdown(int code) {
    finalize();
    std::exit(code);
}

void abort_all(int code) {
#ifdef MRCPP_HAS_MPI
    int ini = 0;
    MPI_Initialized(&ini);
    int fini = 0;
    MPI_Finalized(&fini);
    if (ini && !fini) {
        MPI_Abort(MPI_COMM_WORLD, code);
    }
#endif
    std::exit(code);
}

void barrier() {
#ifdef MRCPP_HAS_MPI
    if (size > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
}

void sum(double &x) {
#ifdef MRCPP_HAS_MPI
    if (size > 1) {
        double y = 0.0;
        MPI_Allreduce(&x, &y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        x = y;
    }
#else
    (void)x;
#endif
}

void sum(int &x) {
#ifdef MRCPP_HAS_MPI
    if (size > 1) {
        int y = 0;
        MPI_Allreduce(&x, &y, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        x = y;
    }
#else
    (void)x;
#endif
}

} // namespace parallel
} // namespace tdse

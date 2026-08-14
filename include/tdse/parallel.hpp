#pragma once

/**
 * Hybrid MPI + OpenMP for NumericalTDSE.
 *
 * MRCPP parallelizes *inside* a FunctionTree with OpenMP (project, apply,
 * multiply, …). It does not domain-decompose a single tree over MPI. MPI is
 * therefore used at the *function* level: whole trees live on a rank and can
 * be sent with mrcpp::send_tree / recv_tree.
 *
 *   OpenMP  — every mode: threads inside MRCPP tree operations.
 *   MPI     — orbital mode: orbitals are round-robin across ranks; density
 *             and observables are reduced. Exact N-body is one tree, so extra
 *             ranks replicate the work (a warning is printed).
 *
 * We initialize MPI ourselves (MPI_THREAD_FUNNELED) and do *not* call
 * mrcpp::mpi::initialize(), which would steal ranks for MRChem's coefficient
 * bank and exit those processes.
 */

#include "MRCPP/MWFunctions"
#include "MRCPP/Parallel"

#ifdef MRCPP_HAS_MPI
#include <mpi.h>
#endif

namespace tdse {
namespace parallel {

/** MPI rank / communicator size (1 if built without MPI). */
inline int rank = 0;
inline int size = 1;
/** OpenMP threads used by MRCPP on this rank. */
inline int nthreads = 1;

inline bool io_rank() { return rank == 0; }

/** Orbital i is owned by this rank (round-robin). */
inline bool owns_orbital(int i) { return size <= 1 || (i % size) == rank; }

void init(int &argc, char **argv);
void configure_threads(int nthreads_requested);
void finalize();
[[noreturn]] void shutdown(int code);
[[noreturn]] void abort_all(int code);
void barrier();
void sum(double &x);
void sum(int &x);

/**
 * In-place sum of a FunctionTree over MPI_COMM_WORLD (reduce to rank 0,
 * then broadcast). No-op with one rank or without MPI.
 */
template <int D>
void sum_tree(mrcpp::FunctionTree<D> &tree, double prec) {
#ifdef MRCPP_HAS_MPI
    if (size <= 1) {
        return;
    }
    const auto &mra = tree.getMRA();
    // send_tree uses tag, tag+chunk+1 and tag+chunk+1001; keep a wide gap.
    constexpr int k_up = 17001;
    constexpr int k_down = 29001;
    if (rank == 0) {
        for (int src = 1; src < size; ++src) {
            mrcpp::FunctionTree<D> part(mra);
            mrcpp::recv_tree(part, src, k_up, MPI_COMM_WORLD);
            mrcpp::FunctionTree<D> total(mra);
            mrcpp::add(prec, total, 1.0, tree, 1.0, part);
            tree.clear();
            total.deep_copy(&tree);
        }
        for (int dst = 1; dst < size; ++dst) {
            mrcpp::send_tree(tree, dst, k_down, MPI_COMM_WORLD);
        }
    } else {
        mrcpp::send_tree(tree, 0, k_up, MPI_COMM_WORLD);
        mrcpp::recv_tree(tree, 0, k_down, MPI_COMM_WORLD);
    }
#else
    (void)tree;
    (void)prec;
#endif
}

} // namespace parallel
} // namespace tdse

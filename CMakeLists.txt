cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(NumericalTDSE
    VERSION 1.0.0
    DESCRIPTION "Adaptive multiwavelet TDSE solver built on MRCPP"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(FetchContent)

option(NUMTDSE_OPENMP "Enable OpenMP parallelization (forwarded to MRCPP)" ON)
option(NUMTDSE_MPI "Enable MPI multi-node parallelization (forwarded to MRCPP)" ON)
option(NUMTDSE_FETCH_MRCPP "Download MRCPP with FetchContent if find_package fails" ON)
set(NUMTDSE_MRCPP_GIT_TAG "77187f0e2a070741032997c777b7f47a39abdac5" CACHE STRING
    "MRCPP Git commit (1.6.0-alpha line with TimeEvolutionOperator)")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# OpenMP (optional, recommended) and MPI (optional, multi-node)
# ---------------------------------------------------------------------------
if(NUMTDSE_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
endif()

if(NUMTDSE_MPI)
    find_program(NUMTDSE_MPICXX NAMES mpicxx mpic++ mpiicpc)
    if(NOT NUMTDSE_MPICXX)
        message(WARNING "NUMTDSE_MPI is ON but mpicxx was not found; building without MPI")
        set(NUMTDSE_MPI OFF CACHE BOOL "Enable MPI multi-node parallelization (forwarded to MRCPP)" FORCE)
    else()
        find_package(MPI COMPONENTS CXX)
        if(NOT MPI_CXX_FOUND)
            message(WARNING "NUMTDSE_MPI is ON but MPI was not found; building without MPI")
            set(NUMTDSE_MPI OFF CACHE BOOL "Enable MPI multi-node parallelization (forwarded to MRCPP)" FORCE)
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# MRCPP: prefer a local install, otherwise fetch 1.6.0-alpha from GitHub.
# TimeEvolutionOperator (free-particle semigroup) lives on this line;
# v1.5.0 does not ship it.
# ---------------------------------------------------------------------------
find_package(MRCPP CONFIG QUIET)

if(NOT MRCPP_FOUND)
    if(NOT NUMTDSE_FETCH_MRCPP)
        message(FATAL_ERROR
            "MRCPP was not found. Install it and pass -DMRCPP_DIR=... "
            "or enable NUMTDSE_FETCH_MRCPP.")
    endif()

    message(STATUS "MRCPP not found via find_package; fetching ${NUMTDSE_MRCPP_GIT_TAG}")

    set(ENABLE_TESTS OFF CACHE BOOL "Disable MRCPP test suite" FORCE)
    set(ENABLE_EXAMPLES OFF CACHE BOOL "Disable MRCPP examples" FORCE)
    set(ENABLE_MPI ${NUMTDSE_MPI} CACHE BOOL "Forward MPI to MRCPP" FORCE)
    set(ENABLE_OPENMP ${NUMTDSE_OPENMP} CACHE BOOL "Forward OpenMP to MRCPP" FORCE)

    FetchContent_Declare(mrcpp
        GIT_REPOSITORY https://github.com/MRChemSoft/mrcpp.git
        GIT_TAG        ${NUMTDSE_MRCPP_GIT_TAG}
        GIT_SHALLOW    OFF
    )
    FetchContent_MakeAvailable(mrcpp)
endif()

if(TARGET MRCPP::mrcpp)
    set(NUMTDSE_MRCPP_TARGET MRCPP::mrcpp)
elseif(TARGET mrcpp)
    set(NUMTDSE_MRCPP_TARGET mrcpp)
else()
    message(FATAL_ERROR "MRCPP was configured but no mrcpp CMake target exists")
endif()

# ---------------------------------------------------------------------------
# Solver executable
# ---------------------------------------------------------------------------
add_executable(tdse
    src/main.cpp
    src/parameters.cpp
    src/input.cpp
    src/parallel.cpp
)

target_include_directories(tdse PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tdse PRIVATE ${NUMTDSE_MRCPP_TARGET})

if(OpenMP_CXX_FOUND)
    target_link_libraries(tdse PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(tdse PRIVATE NUMTDSE_HAS_OPENMP=1)
endif()

if(NUMTDSE_MPI AND MPI_CXX_FOUND)
    target_link_libraries(tdse PRIVATE MPI::MPI_CXX)
    target_compile_definitions(tdse PRIVATE NUMTDSE_HAS_MPI=1)
endif()

target_compile_features(tdse PRIVATE cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tdse PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(tdse PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ---------------------------------------------------------------------------
# Smoke test: two RK4 steps of 1D harmonic oscillator (fast, no TimeEvolution)
# ---------------------------------------------------------------------------
enable_testing()
add_test(NAME tdse_smoke
    COMMAND $<TARGET_FILE:tdse> --smoke
)
set_tests_properties(tdse_smoke PROPERTIES
    TIMEOUT 300
)
add_test(NAME tdse_template
    COMMAND $<TARGET_FILE:tdse> --template
)
add_test(NAME tdse_namelist_smoke
    COMMAND $<TARGET_FILE:tdse> ${CMAKE_SOURCE_DIR}/examples/smoke.in
)
set_tests_properties(tdse_namelist_smoke PROPERTIES
    TIMEOUT 300
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_test(NAME tdse_ground_smoke
    COMMAND $<TARGET_FILE:tdse> ${CMAKE_SOURCE_DIR}/examples/ground_smoke.in
)
set_tests_properties(tdse_ground_smoke PROPERTIES
    TIMEOUT 300
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_test(NAME tdse_itp_smoke
    COMMAND $<TARGET_FILE:tdse> ${CMAKE_SOURCE_DIR}/examples/ground_itp.in
)
set_tests_properties(tdse_itp_smoke PROPERTIES
    TIMEOUT 300
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

if(NUMTDSE_MPI AND MPI_CXX_FOUND AND MPIEXEC_EXECUTABLE)
    set(_numtdse_mpi_preflags)
    if(MPIEXEC_EXECUTABLE MATCHES "openmpi" OR EXISTS "/usr/bin/mpirun.openmpi")
        list(APPEND _numtdse_mpi_preflags --allow-run-as-root --oversubscribe)
    endif()
    add_test(NAME tdse_mpi_smoke
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${_numtdse_mpi_preflags}
                $<TARGET_FILE:tdse> --smoke
    )
    set_tests_properties(tdse_mpi_smoke PROPERTIES
        TIMEOUT 300
        ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_rmaps_base_oversubscribe=true;OMP_NUM_THREADS=1"
    )
    add_test(NAME tdse_mpi_orbitals
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${_numtdse_mpi_preflags}
                $<TARGET_FILE:tdse> ${CMAKE_SOURCE_DIR}/examples/orbitals_smoke.in
    )
    set_tests_properties(tdse_mpi_orbitals PROPERTIES
        TIMEOUT 300
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_rmaps_base_oversubscribe=true;OMP_NUM_THREADS=1"
    )
endif()

message(STATUS "NumericalTDSE: MRCPP target = ${NUMTDSE_MRCPP_TARGET}")
message(STATUS "NumericalTDSE: OpenMP       = ${NUMTDSE_OPENMP}")
message(STATUS "NumericalTDSE: MPI          = ${NUMTDSE_MPI}")

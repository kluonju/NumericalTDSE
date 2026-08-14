# NumericalTDSE user guide

English guide. 中文版：[USER_GUIDE.zh.md](USER_GUIDE.zh.md)

This document teaches you how to build the solver, write a namelist, choose a method, run the examples against closed-form results, and plot the output.

---

## 1. What the code does

NumericalTDSE propagates the time-dependent Schrödinger equation in atomic units

\[
i\,\partial_t\psi = \hat H(t)\,\psi,\qquad
\hat H = -\tfrac12\nabla^2 + V(\mathbf r,t)
\]

on an **adaptive multiwavelet** grid provided by [MRCPP](https://github.com/MRChemSoft/mrcpp). You do not choose a uniform mesh. You choose a precision `prec`; the tree is refined until the local wavelet-norm error is below that threshold (and never deeper than `max_depth`).

Two representations:

| `mode` | Wave function | Typical use |
|---|---|---|
| `exact` | One `FunctionTree<D>` with `D = electrons × dim ≤ 3` | 1D one-electron, 1D two- or three-electron, 2D/3D one-electron |
| `orbital` | Up to 4 orbitals, each `FunctionTree<dim>` | Four electrons, or a cheap mean-field stand-in |

Three time propagators: `krylov` (default), `split` (Strang), `rk4`.

Stationary jobs (`calculation = 'ground'` or `'eigen'`) find the lowest eigenstates of the same \(H\) by Lanczos/Ritz (default) or imaginary-time propagation.

---

## 2. Build

Requirements: C++17, CMake ≥ 3.16, `g++` (this project’s CI uses GNU; the system `c++` may be Clang and fail to link `libstdc++`). Optional: OpenMP, MPI, Python 3 with `matplotlib` for plots.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ -DNUMTDSE_OPENMP=ON -DNUMTDSE_MPI=ON
cmake --build build -j --target tdse
ctest --test-dir build --output-on-failure
```

The binary is `build/bin/tdse`. If MRCPP is already installed, pass `-DMRCPP_DIR=.../share/cmake/MRCPP`.

CLI:

```text
tdse job.in
tdse -i job.in
tdse --template      # commented namelist with every keyword
tdse --smoke         # tiny built-in RK4 run (ctest)
tdse --help
```

---

## 3. First calculation

```bash
./build/bin/tdse examples/harmonic_1d.in
python3 examples/compare_analytic.py harmonic_1d_observables.csv
python3 examples/plot_observables.py harmonic_1d_observables.csv
python3 examples/plot_wavefunction.py harmonic_1d_t0 --analytic ho --x0 1 --omega 1 --t 0
```

Or in one step: `./examples/run_and_plot.sh examples/harmonic_1d.in`.

Ground state of the same oscillator (`E_0 = 1/2`):

```bash
./build/bin/tdse examples/ground_1d.in
python3 examples/compare_analytic.py ground_1d_observables.csv --tol-energy 1e-4 --tol-overlap 0.999
python3 examples/plot_observables.py ground_1d_observables.csv
python3 examples/plot_wavefunction.py ground_1d_n0 --analytic hoeig --n 0 --omega 1
```

Lowest four levels (`E_n = n + 1/2`): `./build/bin/tdse examples/eigen_1d.in`.

### Three electrons in a 1D harmonic oscillator

There are two representations. Start with **orbitals** unless you need a correlated \(N\)-body wave function.

**1. Orbital mode (recommended).** Three 1D orbitals, each a `FunctionTree<1>`. With `lambda = 0` the Hamiltonian is three independent oscillators, so the lowest states are \(\varepsilon_n=\omega(n+1/2)\). Three fermions fill \(n=0,1,2\) and the total energy is \(4.5\,\omega\).

```bash
./build/bin/tdse examples/orbitals_3e_ground.in
python3 examples/compare_analytic.py orbitals_3e_ground_observables.csv --tol-energy 1e-4 --tol-overlap 0.999
```

```fortran
&CONTROL
  calculation = 'ground'
  prefix      = 'orbitals_3e_ground'
/
&SYSTEM
  dim       = 1
  electrons = 3
  mode      = 'orbital'
  trap      = 'harmonic'
  omega     = 1.0
  lambda    = 0.0          ! 0.5 → contact Hartree λρ
/
&TIME
  kinetic = 'bs'           ! smooth bound-state energies
/
&EIGEN
  n_states = 3             ! default is `electrons` in orbital mode
  method   = 'lanczos'
/
```

Real-time: `examples/orbitals_3e.in`. MPI can split the three orbitals: `mpirun -np 3 ./build/bin/tdse examples/orbitals_3e.in`.

**2. Exact \(N\)-body.** One `FunctionTree<3>` for \(\psi(x_1,x_2,x_3)\). This is a 3D adaptive tree — keep `prec` coarse and \(T\) short. `fermion = .true.` loads a Slater determinant of HO orbitals \(n=0,1,2\). Exact mode includes soft-Coulomb \(V_{ee}\) unless you turn it off:

```bash
./build/bin/tdse examples/ho_3e_exact.in
```

```fortran
&SYSTEM
  dim       = 1
  electrons = 3
  mode      = 'exact'
  trap      = 'harmonic'
  fermion   = .true.
  ee        = .false.      ! .true. → 1/sqrt((x_i-x_j)²+a²)
/
```

With `ee = .false.` the Slater is the non-interacting fermionic ground state and \(E=4.5\) is conserved. Ground-state search on this tree is the same namelist with `calculation = 'ground'` (much heavier than the orbital job).

This is a **coherent state** of the 1D harmonic oscillator (`α = ω = 1`, `x0 = 1`):

- dipole \(\mu(t) = \cos(t)\)
- energy \(E = 1\) (theory \(\omega/2 + \tfrac12 \omega^2 x_0^2\))
- overlap \(|\langle\psi_\mathrm{num}|\psi_\mathrm{ana}\rangle|\) printed and stored in the CSV

---

## 4. Units, Hamiltonian, initial data

Atomic units: \(\hbar = m_e = e = 4\pi\epsilon_0 = 1\). Time is in \(\hbar/E_h\), length in Bohr.

Potential pieces (see `include/tdse/analytic.hpp`):

- **Harmonic trap** \(\tfrac12 \omega^2 r^2\)
- **Free** \(V=0\)
- **Soft atom** \(-Z/\sqrt{r^2+a^2}\)
- **Laser** \(-E(t)\,x\) (dipole), \(E(t)=E_0 \sin(\omega_L t)\) times an optional \(\sin^2(\pi t/T)\) envelope
- **Exact N-body 1D** extra electron–electron \(1/\sqrt{(x_i-x_j)^2+a^2}\) (off with `ee = .false.`)
- **Orbital mode** optional contact Hartree \(\lambda\rho(\mathbf r)\)

Initial wave function: a Gaussian \(\psi \propto \exp(-\alpha r^2/2)\) displaced by `x0` along \(x\), with optional boost \(\mathrm{e}^{i k_0 x}\). For exact 1D `fermion = .true.`: two electrons → antisymmetrised pair of Gaussians; three electrons in a harmonic trap → Slater determinant of HO orbitals \(n=0,1,2\).

---

## 5. Namelist input

Free-format, Quantum ESPRESSO style: `&SECTION ... /`. Write only the keys you need; the rest keep defaults. Comments: `!` or `#`. Reals: `1.0d-4`. Booleans: `.true.` / `.false.`. Strings: `'quoted'` or bare.

Sections: `&CONTROL` `&MRA` `&TIME` `&SYSTEM` `&INITIAL` `&LASER` `&OUTPUT` `&PARALLEL` `&EIGEN`  
Aliases: `CTRL`; `GRID`/`NUMERICS` → `MRA`; `PROPAGATOR` → `TIME`; `WAVEFUNCTION`/`PSI` → `INITIAL`; `FIELD` → `LASER`; `IO`/`OUT` → `OUTPUT`; `PARA`/`OMP` → `PARALLEL`; `GROUND`/`GS`/`SCF` → `EIGEN`.

### `&CONTROL`

| Keyword | Default | Meaning |
|---|---|---|
| `calculation` | `'tdse'` | `'tdse'` / `'ground'` / `'eigen'` / `'smoke'` (`smoke` is a tiny RK4 preset; later keys override it) |
| `title` | empty | printed in the log |
| `prefix` | empty | if set and `output=` is omitted → `{prefix}_observables.csv` |
| `printlevel` | 0 | MRCPP verbosity |
| `print_every` | 1 | write a CSV row every N steps |
| `ident_check` | `.true.` | \(\\|I\psi-\psi\\|\) at \(t=0\) |
| `renormalize` | `.false.` | restore \(\\|\psi\\|=1\) after each step |
| `validate_free` | `.false.` | overlap vs analytic free Gaussian |
| `validate_ho` | `.false.` | overlap vs HO coherent state (needs `alpha = omega`) |
| `validate` | — | `'free'` / `'ho'` / `'none'` |

### `&MRA`

| Keyword | Default | Meaning |
|---|---|---|
| `prec` | `1d-4` | adaptive wavelet-norm threshold |
| `order` | 7 | polynomial order \(k\) |
| `max_depth` | 20 | refinement cap |
| `L` | 8 | domain \([-L,L]^D\) |
| `basis` | `'interpolating'` | or `'legendre'` (required for 1D `split` + TEO) |

### `&TIME`

| Keyword | Default | Meaning |
|---|---|---|
| `dt` | 0.02 | time step |
| `T` | 0.40 | final time |
| `propagator` | `'krylov'` | `'krylov'` / `'split'` / `'rk4'` |
| `kinetic` | `'abgv'` | `'abgv'` / `'bs'` / `'dconv'`. Use `'bs'` for smooth bound-state energies |
| `krylov_dim` | 12 | Lanczos subspace size |
| `teo_scale` | 8 | `TimeEvolutionOperator` finest scale |

### `&SYSTEM`

| Keyword | Default | Meaning |
|---|---|---|
| `dim` | 1 | spatial dimension 1, 2, 3 |
| `electrons` | 1 | 1–4 |
| `mode` | `'exact'` | `'exact'` or `'orbital'` |
| `trap` | `'harmonic'` | `'harmonic'` / `'free'` / `'atom'` |
| `omega` | 1 | HO frequency |
| `soft_a` | 1 | soft-Coulomb length |
| `Z` | 1 | nuclear charge |
| `lambda` | 0 | orbital contact \(\lambda\) |
| `fermion` | `.false.` | exact 1D: Slater initial data for 2e or 3e |
| `ee` | `.true.` | exact 1D N-body: include soft-Coulomb \(V_{ee}\) |

### `&INITIAL`

| Keyword | Default | Meaning |
|---|---|---|
| `alpha` | 1 | Gaussian width |
| `x0` | 1 | displacement along \(x\) |
| `k0` | 0 | boost \(p_x\) |

### `&LASER`

| Keyword | Default | Meaning |
|---|---|---|
| `E0` | 0 | field amplitude |
| `omega_L` | 0.5 | laser frequency |
| `envelope` | `.false.` | `.true.` → \(\sin^2(\pi t/T)\) |

### `&OUTPUT`

| Keyword | Default | Meaning |
|---|---|---|
| `output` | `'observables.csv'` | CSV path (sets `output=` explicitly) |
| `plot` | empty | 1D line-plot prefix at \(t=0\) and \(t=T\) |
| `n_plot` | 800 | samples on the line |

### `&PARALLEL`

| Keyword | Default | Meaning |
|---|---|---|
| `nthreads` | 0 | OpenMP threads per MPI rank; 0 → `OMP_NUM_THREADS` |

### `&EIGEN`

Used when `calculation = 'ground'` or `'eigen'`. Keywords are also accepted in `&CONTROL`.

| Keyword | Default | Meaning |
|---|---|---|
| `n_states` | 1 / 4 | lowest eigenstates; default 1 for `ground` (exact), 4 for `eigen`, `electrons` in orbital mode |
| `method` | `'lanczos'` | `'lanczos'` (Ritz of \(H\)) or `'itp'` (imaginary time \(\mathrm{e}^{-H\tau}\)) |
| `conv_thr` | `1d-6` | ITP / SCF: stop when \(\lvert\Delta E\rvert\) is below this |
| `residual` | `0` | \(\lVert(H-E)\psi\rVert\) threshold; `0` → \(50\times\) `prec` |
| `max_iter` | 80 | ITP steps or SCF cycles |

Laser field must be off (`E0 = 0`). For several HO states, use a **non-even** trial (`x0 ≠ 0`); an even Gaussian has no overlap with odd eigenfunctions.

---

## 6. Choosing numerics

- **`prec`**: start at `1d-3` to debug, `1d-4` for demos, `1d-5`–`1d-6` for quantitative work. Finer `prec` → more nodes → slower.
- **`L`**: the packet must stay away from \(\pm L\). Free packets spread and travel; enlarge the box (`free_boost.in` uses `L = 12`).
- **`dt`**: RK4 is not unitary — use a smaller step or `renormalize`. Krylov is closer to unitary. Split + TEO is natural for 1D free kinetic evolution.
- **`propagator`**: Krylov is the default for any dimension. `split` in 1D switches the basis to Legendre automatically.
- **Stationary**: `lanczos` takes the ground state from a Krylov space of \(H\) (increase `krylov_dim` if the overlap is poor); a few heat-kernel imag-time steps then strip high-frequency junk from \(\langle H\rangle\). Higher states use a nodal guess plus the same heat polish. For energies of smooth bound states set `kinetic = 'bs'` (B-spline derivative). `itp` uses `dt` and `T` as imaginary time with a Strang split: the kinetic piece is the MRCPP heat kernel \(\exp((\tau/2)\nabla^2)\). Orbital jobs with \(\lambda\neq 0\) wrap an SCF around the inner solver.

### Ground state and lowest eigenstates

The Hamiltonian is the same as in a TDSE run (trap + optional \(e\)–\(e\), no laser).

- **`calculation = 'ground'`** — lowest state (or `n_states` lowest if you set it).
- **`calculation = 'eigen'`** — several lowest states (default 4 in exact mode).

**Lanczos (default).** Builds a Krylov space of \(H\) from a smooth trial. Because the multiwavelet kinetic operator is not bounded below, the algebraically smallest Ritz value is a spurious mode; the physical state is the Ritz vector that overlaps the trial. A few heat-kernel imag-time steps then smooth high-frequency contamination that would otherwise shift \(\langle H\rangle\). Higher states use a nodal (Hermite) guess and the same heat polish in the orthogonal complement.

On the 1D HO, \(E_n=\omega(n+1/2)\) is recovered systematically if you tighten the MRA **and** use `kinetic = 'bs'`:

| Input | `prec` | order | kinetic | \(\max\|E-E_n\|\) | min overlap |
|---|---|---|---|---|---|
| `ground_1d.in` | \(10^{-4}\) | 7 | bs | \(4.5\times10^{-6}\) | \(0.999999\) |
| `eigen_1d.in` (4 states) | \(10^{-4}\) | 7 | bs | \(1.3\times10^{-5}\) | \(0.999998\) |
| `ground_1d_precise.in` | \(10^{-5}\) | 9 | bs | \(1.3\times10^{-7}\) | \(1-6\times10^{-9}\) |
| `eigen_1d_precise.in` (4 states) | \(10^{-5}\) | 9 | bs | \(1.1\times10^{-5}\) | \(0.999998\) |
| ground, `order=9`, `prec=1d-6` | \(10^{-6}\) | 9 | bs | \(1.8\times10^{-6}\) | \(0.9999997\) |

`kinetic = 'abgv'` does **not** improve \(\langle H\rangle\) when `prec` is tightened: more wavelet scales resolve more spurious kinetic modes. Overlap with \(\psi_n\) can still be \(>0.99\) while \(\langle H\rangle\) is off by \(10^{-2}\). The MW residual \(\lVert(H-E)\psi\rVert/\lVert\psi\rVert\) can stay large even when the overlap and \(\Delta E\) are tight; prefer those two as the accuracy metric.

**Imaginary time.** Substitute \(t=-i\tau\) so \(\partial_\tau\psi=-H\psi\). High-energy components decay; after each step the wave function is renormalized. Excited states are obtained sequentially with Gram–Schmidt. For `n_states = 1` the CSV is an \(E(\tau)\) history (same columns as TDSE). For several states, or for Lanczos, the CSV is a spectrum (`state,energy,residual,…`). The imag-time kinetic step is the heat semigroup \(\exp(-T\tau)=\exp((\tau/2)\nabla^2)\); Krylov/RK4 on the MW Hamiltonian is not used because that operator is unbounded below. Convergence is on \(|\Delta E|\), not the MW residual.

Isotropic HO analytic energy (including 2D/3D degeneracy): \(E=\omega(N+D/2)\) with shell \(N=n_1+\cdots+n_D\). Wave-function overlap is filled for 1D all \(n\), and for \(D>1\) only the ground state.

---

## 7. Parallel runs

OpenMP threads inside each `FunctionTree` (all modes):

```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true
export OMP_PLACES=cores
./build/bin/tdse examples/harmonic_1d.in
```

MPI **does not** split one N-body tree. It round-robins **orbitals**:

```bash
mpirun -np 2 ./build/bin/tdse examples/orbitals_2e.in
mpirun -np 4 ./build/bin/tdse examples/orbitals_4e.in
```

See `examples/parallel.sh`. Rank 0 writes the CSV and plots.

---

## 8. Output files

CSV columns:

```text
t, norm, dipole, energy, nodes_re, nodes_im,
overlap_analytic, dipole_analytic, energy_analytic, residual
```

`dipole_analytic` / `energy_analytic` are filled for **one electron**, `mode = 'exact'`, trap `harmonic` or `free` (energy only if `E0 = 0`). `overlap_analytic` is filled when `validate_free` or `validate_ho` is on. `residual` is used in imaginary-time histories.

Lanczos / multi-state jobs write a **spectrum** instead:

```text
state, energy, residual, overlap_analytic, energy_analytic, norm, dipole, nodes_re, nodes_im
```

1D plots (`plot = 'name'`): TDSE uses `name_t0_*` and `name_tT_*`; eigenstates use `name_n0_*`, `name_n1_*`, … (two columns: \(x\), value).

---

## 9. Plotting tools

Need Python 3. Plots need `pip install matplotlib`. Comparisons need only the stdlib.

```bash
python3 examples/analytic_ref.py              # formula self-check
python3 examples/compare_analytic.py run.csv  # RMS |μ−μ_ana|, |E−E_ana|, min overlap
python3 examples/plot_observables.py run.csv -o obs.png
python3 examples/plot_wavefunction.py name_t0 --analytic ho --x0 1 --omega 1 --t 0
python3 examples/plot_wavefunction.py name_n0 --analytic hoeig --n 0 --omega 1
python3 examples/plot_wavefunction.py name_tT --analytic free --alpha 1 --x0 0 --k0 0 --t 0.2
```

`compare_analytic.py --tol-dipole 1e-3 --tol-energy 1e-3 --tol-overlap 0.999 --tol-residual 1e-3` returns a non-zero exit code on failure (useful in scripts). Spectrum CSVs are detected automatically. Imaginary-time histories (overlap increasing) are judged on the last step.

---

## 10. Examples vs analytic results

Closed forms implemented in `include/tdse/analytic.hpp` and `examples/analytic_ref.py` (atomic units, \(m=1\)):

**Harmonic oscillator** (any Gaussian width; Ehrenfest)

\[
\mu(t)=x_0\cos(\omega t)+(k_0/\omega)\sin(\omega t)
\]

\[
E=\frac{D}{4}\Big(\alpha+\frac{\omega^2}{\alpha}\Big)+\tfrac12\omega^2 x_0^2+\tfrac12 k_0^2
\]

If \(\alpha=\omega\) this is a coherent state and `validate_ho` also reports \(|\langle\psi_\mathrm{num}|\psi_\mathrm{ana}\rangle|\).

**HO eigenstates** (stationary jobs)

\[
E_n=\omega\bigl(n+\tfrac12\bigr)\quad(1\mathrm{D}),\qquad
\psi_n(x)=\frac{1}{\sqrt{2^n n!}}\Bigl(\frac{\omega}{\pi}\Bigr)^{1/4}
\mathrm{e}^{-\omega x^2/2}\,H_n(\sqrt{\omega}\,x)
\]

In \(D\) dimensions the \(k\)-th lowest level uses the Cartesian shell \(N=n_1+\cdots+n_D\), \(E=\omega(N+D/2)\).

**Free particle**

\[
\mu(t)=x_0+k_0 t,\qquad E=\tfrac{D}{4}\alpha+\tfrac12 k_0^2
\]

`validate_free` overlaps with the exact spreading Gaussian (including boost \(k_0\)).

**Driven HO** (CW laser, no envelope, \(\omega_L\neq\omega\))

\[
\ddot x+\omega^2 x=E_0\sin(\omega_L t)
\]

has a closed \(\mu(t)\); energy is **not** conserved.

| Input | What it tests | Analytic |
|---|---|---|
| `smoke.in` / `--smoke` | 2 RK4 steps, \(x_0=0.5\) | \(E=0.625\) |
| `harmonic_1d.in` | 1D coherent state, RK4 | \(\mu=\cos t\), \(E=1\), overlap |
| `harmonic_period.in` | one period \(T=2\pi\) | \(\mu=0.5\cos t\), \(E=0.625\) |
| `harmonic_2d.in` | 2D HO, `FunctionTree<2>` | \(\mu_x=\cos t\), \(E=1.5\) |
| `free_particle.in` | spreading packet | \(\mu=0\), \(E=0.25\), overlap |
| `free_boost.in` | travelling packet | \(\mu=-0.5+t\), \(E=0.75\), overlap |
| `laser_1d.in` | CW dipole drive | Ehrenfest \(\mu(t)\); \(E\) not conserved |
| `split_1d.in` | Strang + TEO | same HO coherent state, short \(T\) |
| `atom_1d.in` | soft-Coulomb well | none — check \(\\|\psi\\|\) |
| `helium_1d.in` | exact 1D 2e | none — check \(\\|\psi\\|\) |
| `orbitals_2e.in` | 2 orbitals + \(\lambda\rho\) | none; MPI-friendly |
| `orbitals_4e.in` | 4 orbitals + \(\lambda\rho\) | none; up to 4 MPI ranks |
| `orbitals_3e_ground.in` | 3 HO orbitals, \(\lambda=0\) | \(\varepsilon=0.5,1.5,2.5\) |
| `orbitals_3e.in` | 3-orbital HO TDSE | none; MPI-friendly |
| `ho_3e_exact.in` | exact 1D 3e HO Slater | \(E=4.5\) if `ee=.false.` |
| `orbitals_smoke.in` | MPI ctest | tiny orbital RK4 |
| `ground_smoke.in` | Lanczos HO ground (ctest) | \(E=0.5\) |
| `ground_1d.in` | HO ground, Lanczos + BS | \(E=0.5\), \(\psi_0\), \(\mu=0\) |
| `eigen_1d.in` | four lowest HO states | \(E_n=n+1/2\), Hermite overlap |
| `ground_1d_precise.in` | tighter HO ground | \(\lvert\Delta E\rvert\sim10^{-7}\) |
| `eigen_1d_precise.in` | tighter four HO states | \(\lvert\Delta E\rvert\sim10^{-5}\) |
| `ground_itp.in` | HO ground by imaginary time | \(E(\tau)\to 0.5\) |
| `ground_atom.in` | 1D soft-Coulomb ground | residual only |
| `orbitals_ground.in` | 2 HO orbitals, \(\lambda=0\) | \(\varepsilon=0.5,1.5\) |
| `helium_ground.in` | exact 1D 2e ground | residual only |

Demo inputs use a coarse grid so they finish quickly. For stationary energies set `kinetic = 'bs'` and lower `prec` to `1d-5`–`1d-6` (see `ground_1d_precise.in`). For real-time work, also shrink `dt`.

---

## 11. Troubleshooting

| Symptom | What to try |
|---|---|
| `cannot find -lstdc++` | configure with `-DCMAKE_CXX_COMPILER=g++` |
| packet hits the box | increase `L` |
| \(\\|\psi\\|\) drifts (RK4) | smaller `dt`, or `renormalize = .true.`, or switch to `krylov` |
| overlap ≪ 1 | tighter `prec`, smaller `dt`, check `alpha = omega` for `validate_ho` |
| `split` is slow to start | TEO construction; keep `T` short in tests |
| MPI does not speed up HO | expected: exact mode is one tree; use OpenMP or `mode = 'orbital'` |
| odd HO states missing | set `x0 ≠ 0` so the trial is not even |
| residual stays large | increase `krylov_dim`, tighten `prec`, enlarge `L`; for energies prefer overlap / \(\Delta E\) |
| \(E\) far from analytic while overlap is high | set `kinetic = 'bs'` for smooth bound states; ABGV \(\langle H\rangle\) does not improve with tighter `prec` |
| ITP energy runs to −∞ | the MW kinetic operator is unbounded below; ITP uses the heat kernel. Rebuild if you still have an old binary |
| empty plots | `pip install matplotlib`; run from the directory that contains the `.csv` / `.line` files |

---

## 12. Limits

- MRCPP instantiates `FunctionTree<D>` only for \(D=1,2,3\). Four-electron exact N-body is not possible; use `mode = 'orbital'`.
- `TimeEvolutionOperator` is 1D Legendre only (`τ = Δt/2` because the library stores \(\exp(i\tau\partial_x^2)=\exp(-i T\Delta t)\)).
- Contact \(\lambda\rho\) is not a Coulomb Hartree potential.
- MPI cannot domain-decompose a single tree.
- Stationary Lanczos finds physical low-lying states as Ritz vectors that overlap the smooth trial; the MW kinetic operator is not bounded below, so the algebraically smallest Ritz value is not used raw.
- Continuum problems (`trap = 'free'`) yield box-discretized modes, not true bound states.

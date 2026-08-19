# Examples

Full tutorial: [English user guide](../docs/USER_GUIDE.md) · [中文使用手册](../docs/USER_GUIDE.zh.md)

Run from the repository root after `cmake --build build --target tdse`:

```bash
./build/bin/tdse examples/harmonic_1d.in
python3 examples/compare_analytic.py harmonic_1d_observables.csv
python3 examples/plot_observables.py harmonic_1d_observables.csv
./examples/run_and_plot.sh examples/free_particle.in
```

Plotting needs `pip install matplotlib`. `compare_analytic.py` uses only the Python standard library.

| File | EN | 中文 | Analytic |
|---|---|---|---|
| `harmonic_1d.in` | 1D HO coherent state | 一维谐振子相干态 | \(\mu=\cos t\), \(E=1\), overlap |
| `harmonic_period.in` | one HO period \(T=2\pi\) | 一个振荡周期 | \(\mu=0.5\cos t\), \(E=0.625\) |
| `harmonic_2d.in` | 2D HO | 二维谐振子 | \(\mu_x=\cos t\), \(E=1.5\) |
| `harmonic_2e2d_smoke.in` | 2e in 2D on `FunctionTree<4>` | 二维双电子 4 维树短跑 | \(E=2\) |
| `harmonic_2e2d.in` | same, Lanczos ground | 同上，Lanczos 基态 | \(E=2\omega\) |
| `free_particle.in` | spreading free Gaussian | 自由高斯扩散 | \(\mu=0\), \(E=\alpha/4\), overlap |
| `free_boost.in` | travelling free Gaussian | 带初动量的自由高斯 | \(\mu=x_0+k_0 t\), \(E=0.75\) |
| `laser_1d.in` | CW dipole-driven HO | 连续激光驱动 | Ehrenfest \(\mu(t)\) |
| `split_1d.in` | Strang + TEO | 劈裂算符 + TEO | HO coherent, short \(T\) |
| `atom_1d.in` | 1D soft-Coulomb atom | 一维软库仑原子 | none (watch \(\\|\psi\\|\) ) |
| `helium_1d.in` | exact 1D two-electron | 精确一维双电子 | none |
| `orbitals_2e.in` | 2 orbitals + \(\lambda\rho\) | 两轨道平均场 | none; MPI ok |
| `orbitals_4e.in` | 4 orbitals + \(\lambda\rho\) | 四轨道 | none; up to 4 ranks |
| `orbitals_3e_ground.in` | 3 HO orbitals, \(\lambda=0\) | 三轨道基态 | \(0.5,1.5,2.5\) |
| `orbitals_3e.in` | 3-orbital HO TDSE | 三轨道实时 | none; MPI ok |
| `ho_3e_exact.in` | exact 1D 3e HO + \(V_{ee}\) | 精确三电子+排斥 | \(E>4.5\) |
| `ho_3e_exact_ground.in` | exact 1D 3e interacting GS (high RAM) | 精确三电子基态 | residual |
| `ground_1d.in` | HO ground (Lanczos, BS) | 谐振子基态 | \(E=0.5\), \(\psi_0\) |
| `eigen_1d.in` | four lowest HO states | 最低四本征态 | \(E_n=n+1/2\) |
| `ground_1d_precise.in` | tighter HO ground | 更高精度基态 | \(\lvert\Delta E\rvert\sim10^{-7}\) |
| `eigen_1d_precise.in` | tighter four HO states | 更高精度四态 | \(\lvert\Delta E\rvert\sim10^{-5}\) |
| `ground_itp.in` | HO ground by ITP | 虚时基态 | \(E(\tau)\to 0.5\) |
| `ground_atom.in` | 1D soft-Coulomb ground | 软库仑基态 | residual |
| `orbitals_ground.in` | 2 HO orbitals, \(\lambda=0\) | 两轨道（无相互作用） | \(0.5, 1.5\) |
| `helium_ground.in` | exact 1D 2e singlet GS | 精确双电子单重态基态 | residual |
| `ho_2e_singlet.in` | 1D HO 2e singlet GS | 一维双电子单重态 | \(E=\omega=1\) |
| `ho_2e_triplet.in` | 1D HO 2e triplet GS | 一维双电子三重态 | \(E=2\omega=2\) |
| `invert_smoke.in` | 2e-1D inversion ctest | 一维双电子反演短跑 | L1 of \(n\) |
| `invert_2e1d.in` | TGK08 1D helium inversion | 一维氦密度反演 | recover \(v_\mathrm{ext}\) |
| `invert_2e2d.in` | 2e in 2D as 1e in 4D | 二维双电子（4 维组态） | recover \(v_\mathrm{ext}(x,y)\) |
| `smoke.in` | ctest namelist | 短跑 | \(E\approx 0.625\) |
| `ground_smoke.in` | Lanczos ctest | 基态短跑 | \(E=0.5\) |
| `orbitals_smoke.in` | MPI ctest | MPI 短跑 | — |

Tools: `plot_observables.py`, `plot_wavefunction.py`, `plot_2e_wavefunction.py`, `plot_inversion.py`, `compare_analytic.py`, `analytic_ref.py`, `run_and_plot.sh`, `parallel.sh`.

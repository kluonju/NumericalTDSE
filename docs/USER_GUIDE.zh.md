# NumericalTDSE 使用手册

English version: [USER_GUIDE.md](USER_GUIDE.md)

本文说明如何编译求解器、编写 namelist、选择算法、把算例与解析解对照，以及如何画图。

---

## 1. 程序做什么

NumericalTDSE 在原子单位下传播含时薛定谔方程

\[
i\,\partial_t\psi = \hat H(t)\,\psi,\qquad
\hat H = -\tfrac12\nabla^2 + V(\mathbf r,t)
\]

网格由 [MRCPP](https://github.com/MRChemSoft/mrcpp) 的**自适应多小波**提供。你不选均匀网格，只选精度 `prec`；程序按局部小波模误差加密，且不超过 `max_depth`。

两种表示：

| `mode` | 波函数 | 典型用途 |
|---|---|---|
| `exact` | 一张 `FunctionTree<D>`，`D = electrons × dim ≤ 3` | 一维 1–3 电子，或二维/三维单电子 |
| `orbital` | 最多 4 个轨道，每个是 `FunctionTree<dim>` | 四电子，或廉价平均场 |

三种传播子：`krylov`（默认）、`split`（Strang）、`rk4`。

---

## 2. 编译

依赖：C++17、CMake ≥ 3.16、`g++`（本仓库用 GNU；系统自带的 `c++` 可能是 Clang，会链不上 `libstdc++`）。可选：OpenMP、MPI、带 `matplotlib` 的 Python 3（画图）。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ -DNUMTDSE_OPENMP=ON -DNUMTDSE_MPI=ON
cmake --build build -j --target tdse
ctest --test-dir build --output-on-failure
```

可执行文件是 `build/bin/tdse`。若已安装 MRCPP，加 `-DMRCPP_DIR=.../share/cmake/MRCPP`。

命令行：

```text
tdse job.in
tdse -i job.in
tdse --template      # 带注释的完整 namelist
tdse --smoke         # 内置短跑（ctest）
tdse --help
```

---

## 3. 第一次计算

```bash
./build/bin/tdse examples/harmonic_1d.in
python3 examples/compare_analytic.py harmonic_1d_observables.csv
python3 examples/plot_observables.py harmonic_1d_observables.csv
python3 examples/plot_wavefunction.py harmonic_1d_t0 --analytic ho --x0 1 --omega 1 --t 0
```

也可以一步完成：`./examples/run_and_plot.sh examples/harmonic_1d.in`。

这是一维谐振子的**相干态**（`α = ω = 1`，`x0 = 1`）：

- 偶极 \(\mu(t) = \cos(t)\)
- 能量 \(E = 1\)（理论 \(\omega/2 + \tfrac12 \omega^2 x_0^2\)）
- 重叠 \(|\langle\psi_\mathrm{num}|\psi_\mathrm{ana}\rangle|\) 会打印并写入 CSV

---

## 4. 单位、哈密顿量、初态

原子单位：\(\hbar = m_e = e = 4\pi\epsilon_0 = 1\)。时间单位 \(\hbar/E_h\)，长度单位 Bohr。

势能（见 `include/tdse/analytic.hpp`）：

- **谐振子** \(\tfrac12 \omega^2 r^2\)
- **自由** \(V=0\)
- **软原子** \(-Z/\sqrt{r^2+a^2}\)
- **激光** \(-E(t)\,x\)（偶极），\(E(t)=E_0 \sin(\omega_L t)\)，可选包络 \(\sin^2(\pi t/T)\)
- **精确一维 N 体** 额外电子–电子 \(1/\sqrt{(x_i-x_j)^2+a^2}\)
- **轨道模式** 可选接触 Hartree \(\lambda\rho(\mathbf r)\)

初态：高斯 \(\psi \propto \exp(-\alpha r^2/2)\)，沿 \(x\) 位移 `x0`，可选 boost \(\mathrm{e}^{i k_0 x}\)。一维双电子且 `fermion = .true.` 时初态反对称化。

---

## 5. Namelist 输入

自由格式，Quantum ESPRESSO 风格：`&SECTION ... /`。只写要改的关键字，其余用默认。注释：`!` 或 `#`。实数：`1.0d-4`。逻辑：`.true.` / `.false.`。字符串：`'quoted'` 或裸词。

段名：`&CONTROL` `&MRA` `&TIME` `&SYSTEM` `&INITIAL` `&LASER` `&OUTPUT` `&PARALLEL`  
别名：`CTRL`；`GRID`/`NUMERICS` → `MRA`；`PROPAGATOR` → `TIME`；`WAVEFUNCTION`/`PSI` → `INITIAL`；`FIELD` → `LASER`；`IO`/`OUT` → `OUTPUT`；`PARA`/`OMP` → `PARALLEL`。

### `&CONTROL`

| 关键字 | 默认 | 含义 |
|---|---|---|
| `calculation` | `'tdse'` | `'tdse'` 或 `'smoke'`（短跑预设，后面的关键字仍生效） |
| `title` | 空 | 写进日志 |
| `prefix` | 空 | 未写 `output=` 时观测文件为 `{prefix}_observables.csv` |
| `printlevel` | 0 | MRCPP 详细程度 |
| `print_every` | 1 | 每 N 步写一行 CSV |
| `ident_check` | `.true.` | \(t=0\) 时 \(\\|I\psi-\psi\\|\) |
| `renormalize` | `.false.` | 每步把 \(\\|\psi\\|\) 拉回 1 |
| `validate_free` | `.false.` | 与解析自由高斯重叠 |
| `validate_ho` | `.false.` | 与谐振子相干态重叠（需要 `alpha = omega`） |
| `validate` | — | `'free'` / `'ho'` / `'none'` |

### `&MRA`

| 关键字 | 默认 | 含义 |
|---|---|---|
| `prec` | `1d-4` | 自适应小波模阈值 |
| `order` | 7 | 多项式阶 \(k\) |
| `max_depth` | 20 | 加密上限 |
| `L` | 8 | 盒子 \([-L,L]^D\) |
| `basis` | `'interpolating'` | 或 `'legendre'`（1D `split` + TEO 需要） |

### `&TIME`

| 关键字 | 默认 | 含义 |
|---|---|---|
| `dt` | 0.02 | 时间步 |
| `T` | 0.40 | 终止时间 |
| `propagator` | `'krylov'` | `'krylov'` / `'split'` / `'rk4'` |
| `kinetic` | `'abgv'` | `'abgv'` / `'bs'` / `'dconv'` |
| `krylov_dim` | 12 | Lanczos 子空间维数 |
| `teo_scale` | 8 | `TimeEvolutionOperator` 最细尺度 |

### `&SYSTEM`

| 关键字 | 默认 | 含义 |
|---|---|---|
| `dim` | 1 | 空间维数 1、2、3 |
| `electrons` | 1 | 1–4 |
| `mode` | `'exact'` | `'exact'` 或 `'orbital'` |
| `trap` | `'harmonic'` | `'harmonic'` / `'free'` / `'atom'` |
| `omega` | 1 | 谐振子频率 |
| `soft_a` | 1 | 软库仑长度 |
| `Z` | 1 | 核电荷 |
| `lambda` | 0 | 轨道接触 \(\lambda\) |
| `fermion` | `.false.` | 精确 2e–1D 初态反对称化 |

### `&INITIAL`

| 关键字 | 默认 | 含义 |
|---|---|---|
| `alpha` | 1 | 高斯宽度 |
| `x0` | 1 | 沿 \(x\) 的位移 |
| `k0` | 0 | boost \(p_x\) |

### `&LASER`

| 关键字 | 默认 | 含义 |
|---|---|---|
| `E0` | 0 | 场强 |
| `omega_L` | 0.5 | 激光频率 |
| `envelope` | `.false.` | `.true.` → \(\sin^2(\pi t/T)\) |

### `&OUTPUT`

| 关键字 | 默认 | 含义 |
|---|---|---|
| `output` | `'observables.csv'` | CSV 路径（显式指定） |
| `plot` | 空 | \(t=0\) 与 \(t=T\) 的一维线框图前缀 |
| `n_plot` | 800 | 线上采样点数 |

### `&PARALLEL`

| 关键字 | 默认 | 含义 |
|---|---|---|
| `nthreads` | 0 | 每个 MPI rank 的 OpenMP 线程；0 → `OMP_NUM_THREADS` |

---

## 6. 如何选数值参数

- **`prec`**：调试用 `1d-3`，演示用 `1d-4`，定量用 `1d-5`–`1d-6`。更小的 `prec` → 更多节点 → 更慢。
- **`L`**：波包不要碰到 \(\pm L\)。自由高斯会扩散、会飞走，盒子要加大（`free_boost.in` 用 `L = 12`）。
- **`dt`**：RK4 不正交，步长宜小，或开 `renormalize`。Krylov 更接近幺正。1D 动能用 `split` + TEO 很自然。
- **`propagator`**：任意维默认 Krylov。1D `split` 会自动改用 Legendre 基。

---

## 7. 并行

所有模式都可以在一张 `FunctionTree` **内部**用 OpenMP：

```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true
export OMP_PLACES=cores
./build/bin/tdse examples/harmonic_1d.in
```

MPI **不会**把一张 N 体树按空间切开，而是把**轨道** round-robin 分到各 rank：

```bash
mpirun -np 2 ./build/bin/tdse examples/orbitals_2e.in
mpirun -np 4 ./build/bin/tdse examples/orbitals_4e.in
```

见 `examples/parallel.sh`。只有 rank 0 写 CSV 和线图。

---

## 8. 输出文件

CSV 列：

```text
t, norm, dipole, energy, nodes_re, nodes_im,
overlap_analytic, dipole_analytic, energy_analytic
```

`dipole_analytic` / `energy_analytic` 在**单电子**、`mode = 'exact'`、势阱为 `harmonic` 或 `free` 时填写（能量还要求 `E0 = 0`）。`overlap_analytic` 在打开 `validate_free` 或 `validate_ho` 时填写。

一维图（`plot = 'name'`）：`name_t0_re.line`、`name_t0_im.line`、`name_tT_re.line`、`name_tT_im.line`（两列：\(x\) 与函数值）。

---

## 9. 画图工具

需要 Python 3。图需要 `pip install matplotlib`。数值对照只用标准库。

```bash
python3 examples/analytic_ref.py              # 公式自检
python3 examples/compare_analytic.py run.csv  # |μ−μ_ana|、|E−E_ana| 的 RMS，以及最小重叠
python3 examples/plot_observables.py run.csv -o obs.png
python3 examples/plot_wavefunction.py name_t0 --analytic ho --x0 1 --omega 1 --t 0
python3 examples/plot_wavefunction.py name_tT --analytic free --alpha 1 --x0 0 --k0 0 --t 0.2
```

`compare_analytic.py --tol-dipole 1e-3 --tol-energy 1e-3 --tol-overlap 0.999` 超差时返回非零退出码，便于脚本检查。

---

## 10. 算例与解析解

封闭公式在 `include/tdse/analytic.hpp` 与 `examples/analytic_ref.py`（原子单位，\(m=1\)）：

**谐振子**（任意高斯宽度；Ehrenfest）

\[
\mu(t)=x_0\cos(\omega t)+(k_0/\omega)\sin(\omega t)
\]

\[
E=\frac{D}{4}\Big(\alpha+\frac{\omega^2}{\alpha}\Big)+\tfrac12\omega^2 x_0^2+\tfrac12 k_0^2
\]

若 \(\alpha=\omega\) 则是相干态，`validate_ho` 还会给出 \(|\langle\psi_\mathrm{num}|\psi_\mathrm{ana}\rangle|\)。

**自由粒子**

\[
\mu(t)=x_0+k_0 t,\qquad E=\tfrac{D}{4}\alpha+\tfrac12 k_0^2
\]

`validate_free` 与精确扩散高斯（含 boost \(k_0\)）做重叠。

**驱动谐振子**（连续激光、无包络，\(\omega_L\neq\omega\)）

\[
\ddot x+\omega^2 x=E_0\sin(\omega_L t)
\]

有封闭的 \(\mu(t)\)；**能量不守恒**。

| 输入 | 测什么 | 解析 |
|---|---|---|
| `smoke.in` / `--smoke` | 两步 RK4，\(x_0=0.5\) | \(E=0.625\) |
| `harmonic_1d.in` | 1D 相干态，RK4 | \(\mu=\cos t\)，\(E=1\)，重叠 |
| `harmonic_period.in` | 一个周期 \(T=2\pi\) | \(\mu=0.5\cos t\)，\(E=0.625\) |
| `harmonic_2d.in` | 2D 谐振子，`FunctionTree<2>` | \(\mu_x=\cos t\)，\(E=1.5\) |
| `free_particle.in` | 扩散波包 | \(\mu=0\)，\(E=0.25\)，重叠 |
| `free_boost.in` | 飞行波包 | \(\mu=-0.5+t\)，\(E=0.75\)，重叠 |
| `laser_1d.in` | 连续偶极驱动 | Ehrenfest \(\mu(t)\)；\(E\) 不守恒 |
| `split_1d.in` | Strang + TEO | 同上相干态，短 \(T\) |
| `atom_1d.in` | 软库仑阱 | 无 — 看 \(\\|\psi\\|\) |
| `helium_1d.in` | 精确 1D 双电子 | 无 — 看 \(\\|\psi\\|\) |
| `orbitals_2e.in` | 两轨道 + \(\lambda\rho\) | 无；适合 MPI |
| `orbitals_4e.in` | 四轨道 + \(\lambda\rho\) | 无；最多 4 个 MPI rank |
| `orbitals_smoke.in` | MPI ctest | 轨道短跑 |

演示输入网格较粗，为了尽快跑完。正式计算请把 `prec` 降到 `1d-5`–`1d-6` 并减小 `dt`。

---

## 11. 排错

| 现象 | 尝试 |
|---|---|
| `cannot find -lstdc++` | 配置时加 `-DCMAKE_CXX_COMPILER=g++` |
| 波包碰到盒子 | 加大 `L` |
| RK4 下 \(\\|\psi\\|\) 漂移 | 减小 `dt`，或 `renormalize = .true.`，或改用 `krylov` |
| 重叠远小于 1 | 更小 `prec` / `dt`；`validate_ho` 时检查 `alpha = omega` |
| `split` 启动很慢 | 在构造 TEO；测试时 `T` 保持很短 |
| MPI 不能加速谐振子 | 正常：精确模式只有一棵树；用 OpenMP 或 `mode = 'orbital'` |
| 画不出图 | `pip install matplotlib`；在含 `.csv` / `.line` 的目录下运行 |

---

## 12. 限制

- MRCPP 只实例化 \(D=1,2,3\) 的 `FunctionTree<D>`。四电子精确 N 体做不到，请用 `mode = 'orbital'`。
- `TimeEvolutionOperator` 仅 1D Legendre（传入 \(\tau=\Delta t/2\)，因为库里是 \(\exp(i\tau\partial_x^2)=\exp(-i T\Delta t)\)）。
- 接触 \(\lambda\rho\) 不是库仑 Hartree。
- MPI 不能把单棵树按空间分解。

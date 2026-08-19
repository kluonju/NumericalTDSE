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
| `exact` | 一张 `FunctionTree<D>`，`D = electrons × dim ≤ 4` | 一维 1–3 电子，二维/三维单电子，或二维双电子 |
| `orbital` | 最多 4 个轨道，每个是 `FunctionTree<dim>` | 四电子，或廉价平均场 |

三种传播子：`krylov`（默认）、`split`（Strang）、`rk4`。

定态计算（`calculation = 'ground'` 或 `'eigen'`）用 Lanczos/Ritz（默认）或虚时传播求同一 \(H\) 的最低本征态。`calculation = 'invert'` 对两电子做相互作用密度反演 \(v_\mathrm{ext}[n]\)（TGK08）。

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

同一谐振子的基态（\(E_0 = 1/2\)）：

```bash
./build/bin/tdse examples/ground_1d.in
python3 examples/compare_analytic.py ground_1d_observables.csv --tol-energy 1e-4 --tol-overlap 0.999
python3 examples/plot_observables.py ground_1d_observables.csv
python3 examples/plot_wavefunction.py ground_1d_n0 --analytic hoeig --n 0 --omega 1
```

最低四态（\(E_n = n + 1/2\)）：`./build/bin/tdse examples/eigen_1d.in`。

### 一维谐振子里的三个电子

两种表示。一般先用**轨道模式**；只有需要关联 \(N\) 体波函数时才上精确树。

**1. 轨道模式（推荐）。** 三个一维轨道，每个是 `FunctionTree<1>`。`lambda = 0` 时是三个独立振子，最低轨道能 \(\varepsilon_n=\omega(n+1/2)\)。三个费米子占据 \(n=0,1,2\)，总能量 \(4.5\,\omega\)。

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
  lambda    = 0.0          ! 0.5 → 接触 Hartree λρ
/
&TIME
  kinetic = 'bs'
/
&EIGEN
  n_states = 3
  method   = 'lanczos'
/
```

实时：`examples/orbitals_3e.in`。MPI 可按轨道分：`mpirun -np 3 ./build/bin/tdse examples/orbitals_3e.in`。

**2. 精确 \(N\) 体（打开软库仑 \(V_{ee}\)）。** 一张 `FunctionTree<3>` 表示 \(\psi(x_1,x_2,x_3)\)。坐标 `r[i]` 是第 \(i\) 个电子。哈密顿量是

\[
H = \sum_{i=1}^{3}\Bigl(-\tfrac12\partial_{x_i}^2 + \tfrac12\omega^2 x_i^2\Bigr)
+ \sum_{i<j}\frac{1}{\sqrt{(x_i-x_j)^2+a^2}}.
\]

这是三维自适应树，`prec` 要粗、\(T\) 要短。`fermion = .true.` 用 HO 轨道 \(n=0,1,2\) 的 Slater 行列式（无相互作用基态，也是有相互作用时的好初猜）。没有初等解析能；排斥使 \(E\) 高于 \(4.5\,\omega\)。

```bash
./build/bin/tdse examples/ho_3e_exact.in            # 短时 TDSE（推荐）
./build/bin/tdse examples/ho_3e_exact_ground.in     # 可选 Lanczos；三维树，占内存
```

```fortran
&SYSTEM
  dim       = 1
  electrons = 3
  mode      = 'exact'
  trap      = 'harmonic'
  omega     = 1.0
  fermion   = .true.
  ee        = .true.
  soft_a    = 1.0          ! V_ee 里的正则化长度 a
/
```

只有做无相互作用对照时才设 `ee = .false.`（\(E=4.5\) 守恒）。轨道模式 `lambda > 0` 是接触平均场，和这里的 \(V_{ee}\) 不是一回事。

---

## 4. 单位、哈密顿量、初态

原子单位：\(\hbar = m_e = e = 4\pi\epsilon_0 = 1\)。时间单位 \(\hbar/E_h\)，长度单位 Bohr。

势能（见 `include/tdse/analytic.hpp`）：

- **谐振子** \(\tfrac12 \omega^2 r^2\)
- **自由** \(V=0\)
- **软原子** \(-Z/\sqrt{r^2+a^2}\)
- **激光** \(-E(t)\,x\)（偶极），\(E(t)=E_0 \sin(\omega_L t)\)，可选包络 \(\sin^2(\pi t/T)\)
- **精确一维 N 体** 额外电子–电子 \(1/\sqrt{(x_i-x_j)^2+a^2}\)（`ee = .false.` 可关）
- **轨道模式** 可选接触 Hartree \(\lambda\rho(\mathbf r)\)

初态：高斯 \(\psi \propto \exp(-\alpha r^2/2)\)，沿 \(x\) 位移 `x0`，可选 boost \(\mathrm{e}^{i k_0 x}\)。精确双电子用 `spin = 'singlet'`（空间对称）或 `'triplet'`（空间反对称）；`fermion = .true.` 在两电子时等于三重态。三电子谐振子 `fermion = .true.`：HO 轨道 \(n=0,1,2\) 的 Slater 行列式。无相互作用 1D HO：单重态 \(E=\omega\)，三重态 \(E=2\omega\)。

---

## 5. Namelist 输入

自由格式，Quantum ESPRESSO 风格：`&SECTION ... /`。只写要改的关键字，其余用默认。注释：`!` 或 `#`。实数：`1.0d-4`。逻辑：`.true.` / `.false.`。字符串：`'quoted'` 或裸词。

段名：`&CONTROL` `&MRA` `&TIME` `&SYSTEM` `&INITIAL` `&LASER` `&OUTPUT` `&PARALLEL` `&EIGEN` `&INVERT`  
别名：`CTRL`；`GRID`/`NUMERICS` → `MRA`；`PROPAGATOR` → `TIME`；`WAVEFUNCTION`/`PSI` → `INITIAL`；`FIELD` → `LASER`；`IO`/`OUT` → `OUTPUT`；`PARA`/`OMP` → `PARALLEL`；`GROUND`/`GS`/`SCF` → `EIGEN`；`TGK`/`PEIRS`/`INVERSION` → `INVERT`。

### `&CONTROL`

| 关键字 | 默认 | 含义 |
|---|---|---|
| `calculation` | `'tdse'` | `'tdse'` / `'ground'` / `'eigen'` / `'invert'` / `'smoke'`（`smoke` 是短跑预设，后面的关键字仍生效） |
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
| `kinetic` | `'abgv'` | `'abgv'` / `'bs'` / `'dconv'`。求平滑束缚态能量时用 `'bs'` |
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
| `fermion` | `.false.` | 精确一维 2e/3e Slater 初态；2e 等价于三重态 |
| `spin` | 未指定 | 精确双电子：`'singlet'`（空间偶）或 `'triplet'`（空间奇） |
| `ee` | `.true.` | 精确一维 N 体是否含软库仑 \(V_{ee}\) |

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

### `&EIGEN`

在 `calculation = 'ground'` 或 `'eigen'` 时使用。关键字也可写在 `&CONTROL` 里。

| 关键字 | 默认 | 含义 |
|---|---|---|
| `n_states` | 1 / 4 | 最低本征态个数；`ground` 精确模式默认为 1，`eigen` 为 4，轨道模式为 `electrons` |
| `method` | `'lanczos'` | `'lanczos'`（\(H\) 的 Ritz）或 `'itp'`（虚时 \(\mathrm{e}^{-H\tau}\)） |
| `conv_thr` | `1d-6` | ITP / SCF：\(\lvert\Delta E\rvert\) 阈值 |
| `residual` | `0` | \(\lVert(H-E)\psi\rVert\) 阈值；`0` → \(50\times\) `prec` |
| `max_iter` | 80 | 虚时步数或 SCF 循环 |

### `&INVERT`

`calculation = 'invert'` 时使用。两电子相互作用基态的密度反演 \(n(\mathbf r)\to v_\mathrm{ext}(\mathbf r)\)，算法来自 Thiele, Gross, Kümmel, Phys. Rev. Lett. **100**, 153004 (2008) 以及 Peirs–Van Neck–Waroquier 迭代。关键字也可写在 `&CONTROL` 里。

| 关键字 | 默认 | 含义 |
|---|---|---|
| `target` | `'self'` | `'self'`：先用 namelist 势阱求基态密度再反演；`'file'`：读 `density_file` |
| `guess` | `'scaled'` | `'scaled'` / `'harmonic'` / `'zero'` / `'atom'` / `'hx'`（\(v_s-\tfrac12 v_H\)） |
| `n_grid` | 49 / 15 | 每个空间方向的格点数（强制为奇数）。1D 默认 49，2D 默认 15 |
| `gamma` | 0.25 | 步长 \(\gamma\)：\(v\leftarrow v+\gamma(w_0+\lvert r\rvert^\beta)(n-n^*)\) |
| `beta` | 1 | 尾部权重 \(\beta\) |
| `w0` | 1 | 权重下限；`0` 时冻结 \(v(0)\)（与论文一致） |
| `tol` | `1d-4` | \(\int\lvert n-n^*\rvert\) 小于此值则停止 |
| `maxiter` | 40 | 外层迭代 |
| `inner` | 40 | 每次外层迭代的虚时步数 |
| `tau` | 0.08 | 虚时步长（隐式动能劈裂） |
| `scale` | 0.55 | 缩放势阱初猜 |
| `ncut` | `1d-3` | 计算 \(v_s\) 和 \(v\) RMS 时的密度阈值 |
| `check` | `.false.` | L1 误差不下降则非零退出 |
| `ks_only` | `.false.` | 跳过相互作用反演，只输出 \(v_s,v_H,v_c\) |
| `density_file` | 空 | `target = 'file'` 时的目标密度 |

**用 4 维单粒子，而不是“二维套二维”。** 两个二维电子的波函数是 \(\psi(x_1,y_1,x_2,y_2)\)，这就是 4 维组态空间里的**一条**薛定谔方程：动能是 4 维拉普拉斯，\(v(r_1)+v(r_2)+W(\lvert r_1-r_2\rvert)\) 是局域 4 维势。若在二维轨道基里做 CI（“2D in 2D”），需要双电子积分，而且基组截断。Fetch 下来的 MRCPP 会在本地打补丁（`cmake/patch_mrcpp_d4.py`），因此精确 TDSE / 基态可以用 `FunctionTree<4>`（`examples/harmonic_2e2d.in`）。反演仍走均匀网格。反演的对象仍是物理的 \(v_\mathrm{ext}(x,y)\)。论文里的一维氦模型同理，只是 \(\psi(x_1,x_2)\) 在 \(N\times N\) 格子上。

得到 \(v_\mathrm{ext}\) 后，两电子单态的 KS 反演是代数的：\(\varphi=\sqrt{n/2}\)，\(v_s=(2\varphi)^{-1}\nabla^2\varphi+\mathrm{const}\)，\(v_c=v_s-\tfrac12 v_H-v_\mathrm{ext}\)。

```bash
./build/bin/tdse examples/invert_2e1d.in
./build/bin/tdse examples/invert_2e2d.in
python3 examples/plot_inversion.py invert_2e1d_observables.csv
```

激光必须关闭（`E0 = 0`）。要求多个谐振子态时，初猜不要取偶函数（`x0 ≠ 0`），否则与奇宇称本征函数正交。

---

## 6. 如何选数值参数

- **`prec`**：调试用 `1d-3`，演示用 `1d-4`，定量用 `1d-5`–`1d-6`。更小的 `prec` → 更多节点 → 更慢。
- **`L`**：波包不要碰到 \(\pm L\)。自由高斯会扩散、会飞走，盒子要加大（`free_boost.in` 用 `L = 12`）。
- **`dt`**：RK4 不正交，步长宜小，或开 `renormalize`。Krylov 更接近幺正。1D 动能用 `split` + TEO 很自然。
- **`propagator`**：任意维默认 Krylov。1D `split` 会自动改用 Legendre 基。
- **定态**：`lanczos` 从 \(H\) 的 Krylov 空间取基态（重叠不够就加大 `krylov_dim`），再用几步热核虚时去掉高频污染。激发态用带节点的初猜加同样的热核打磨。平滑束缚态能量请设 `kinetic = 'bs'`（B-spline 导数）。`itp` 把 `dt`、`T` 当作虚时，用 Strang 劈裂：动能一步是 MRCPP 热核 \(\exp((\tau/2)\nabla^2)\)。轨道模式且 \(\lambda\neq 0\) 时，外层再套一层 SCF。

### 基态与最低本征态

哈密顿量与 TDSE 相同（势阱 + 可选电子–电子，无激光）。

- **`calculation = 'ground'`** — 最低态（若写了 `n_states` 则求最低若干态）。
- **`calculation = 'eigen'`** — 若干最低态（精确模式默认 4 个）。

**Lanczos（默认）。** 从光滑初猜建 \(H\) 的 Krylov 空间。多小波动能算子没有下界，代数上最小的 Ritz 值是虚假模；物理态是与初猜重叠最大的 Ritz 向量。随后用几步热核虚时去掉会拉动 \(\langle H\rangle\) 的高频成分。更高的态用带节点的（Hermite）初猜，并在正交补里做同样的热核打磨。

一维谐振子 \(E_n=\omega(n+1/2)\) 可以系统逼近，但需要收紧 MRA **并且** 使用 `kinetic = 'bs'`：

| 输入 | `prec` | order | kinetic | \(\max\|E-E_n\|\) | 最小重叠 |
|---|---|---|---|---|---|
| `ground_1d.in` | \(10^{-4}\) | 7 | bs | \(4.5\times10^{-6}\) | \(0.999999\) |
| `eigen_1d.in`（4 态） | \(10^{-4}\) | 7 | bs | \(1.3\times10^{-5}\) | \(0.999998\) |
| `ground_1d_precise.in` | \(10^{-5}\) | 9 | bs | \(1.3\times10^{-7}\) | \(1-6\times10^{-9}\) |
| `eigen_1d_precise.in`（4 态） | \(10^{-5}\) | 9 | bs | \(1.1\times10^{-5}\) | \(0.999998\) |
| 基态，`order=9`，`prec=1d-6` | \(10^{-6}\) | 9 | bs | \(1.8\times10^{-6}\) | \(0.9999997\) |

`kinetic = 'abgv'` 把 `prec` 收紧时 \(\langle H\rangle\) **不会**变好：更多小波尺度会解析出更多虚假动能模。与 \(\psi_n\) 的重叠仍可 \(>0.99\)，但 \(\langle H\rangle\) 能差 \(10^{-2}\)。MW 残差即使重叠和 \(\Delta E\) 已经很好也可能偏大；精度请看后两者。

**虚时。** 令 \(t=-i\tau\)，则 \(\partial_\tau\psi=-H\psi\)。高能成分衰减，每步归一化。激发态按顺序求，并用 Gram–Schmidt 正交。`n_states = 1` 时 CSV 是 \(E(\tau)\) 历史（列与 TDSE 相同）。多态或 Lanczos 写出谱文件（`state,energy,residual,…`）。虚时动能用热半群 \(\exp(-T\tau)=\exp((\tau/2)\nabla^2)\)；不用 MW 哈密顿的 Krylov/RK4，因为那个算符没有下界。收敛看 \(|\Delta E|\)，不以 MW 残差为准。

各向同性谐振子解析能（含 2D/3D 简并）：\(E=\omega(N+D/2)\)，壳层 \(N=n_1+\cdots+n_D\)。波函数重叠：一维对所有 \(n\) 填写，\(D>1\) 只填基态。

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
overlap_analytic, dipole_analytic, energy_analytic, residual
```

`dipole_analytic` / `energy_analytic` 在**单电子**、`mode = 'exact'`、势阱为 `harmonic` 或 `free` 时填写（能量还要求 `E0 = 0`）。`overlap_analytic` 在打开 `validate_free` 或 `validate_ho` 时填写。`residual` 用于虚时历史。

Lanczos / 多态任务改为写**谱**：

```text
state, energy, residual, overlap_analytic, energy_analytic, norm, dipole, nodes_re, nodes_im
```

一维图（`plot = 'name'`）：TDSE 用 `name_t0_*`、`name_tT_*`；本征态用 `name_n0_*`、`name_n1_*`、…（两列：\(x\) 与函数值）。

---

## 9. 画图工具

需要 Python 3。图需要 `pip install matplotlib`。数值对照只用标准库。

```bash
python3 examples/analytic_ref.py              # 公式自检
python3 examples/compare_analytic.py run.csv  # |μ−μ_ana|、|E−E_ana| 的 RMS，以及最小重叠
python3 examples/plot_observables.py run.csv -o obs.png
python3 examples/plot_wavefunction.py name_t0 --analytic ho --x0 1 --omega 1 --t 0
python3 examples/plot_wavefunction.py name_n0 --analytic hoeig --n 0 --omega 1
python3 examples/plot_wavefunction.py name_tT --analytic free --alpha 1 --x0 0 --k0 0 --t 0.2
```

`compare_analytic.py --tol-dipole 1e-3 --tol-energy 1e-3 --tol-overlap 0.999 --tol-residual 1e-3` 超差时返回非零退出码，便于脚本检查。谱 CSV 会自动识别。虚时历史（重叠上升）按最后一步检查。

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

**谐振子本征态**（定态计算）

\[
E_n=\omega\bigl(n+\tfrac12\bigr)\quad(1\mathrm{D}),\qquad
\psi_n(x)=\frac{1}{\sqrt{2^n n!}}\Bigl(\frac{\omega}{\pi}\Bigr)^{1/4}
\mathrm{e}^{-\omega x^2/2}\,H_n(\sqrt{\omega}\,x)
\]

\(D\) 维时第 \(k\) 个最低能级用直角坐标壳层 \(N=n_1+\cdots+n_D\)，\(E=\omega(N+D/2)\)。

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
| `harmonic_2e2d_smoke.in` | 二维双电子，`FunctionTree<4>` 短跑 | \(E=2\) |
| `harmonic_2e2d.in` | 同上，Lanczos 基态 | 非相互作用 \(E=2\omega\) |
| `free_particle.in` | 扩散波包 | \(\mu=0\)，\(E=0.25\)，重叠 |
| `free_boost.in` | 飞行波包 | \(\mu=-0.5+t\)，\(E=0.75\)，重叠 |
| `laser_1d.in` | 连续偶极驱动 | Ehrenfest \(\mu(t)\)；\(E\) 不守恒 |
| `split_1d.in` | Strang + TEO | 同上相干态，短 \(T\) |
| `atom_1d.in` | 软库仑阱 | 无 — 看 \(\\|\psi\\|\) |
| `helium_1d.in` | 精确 1D 双电子 | 无 — 看 \(\\|\psi\\|\) |
| `orbitals_2e.in` | 两轨道 + \(\lambda\rho\) | 无；适合 MPI |
| `orbitals_4e.in` | 四轨道 + \(\lambda\rho\) | 无；最多 4 个 MPI rank |
| `orbitals_3e_ground.in` | 三谐振子轨道，\(\lambda=0\) | \(\varepsilon=0.5,1.5,2.5\) |
| `orbitals_3e.in` | 三轨道实时 | 无；适合 MPI |
| `ho_3e_exact.in` | 精确一维三电子 + \(V_{ee}\) | 看 \(\\|\psi\\|\)，\(E>4.5\) |
| `ho_3e_exact_ground.in` | 精确三电子相互作用基态（占内存） | 残差；\(E>4.5\) |
| `orbitals_smoke.in` | MPI ctest | 轨道短跑 |
| `ground_smoke.in` | Lanczos 谐振子基态（ctest） | \(E=0.5\) |
| `ground_1d.in` | 谐振子基态，Lanczos + BS | \(E=0.5\)，\(\psi_0\)，\(\mu=0\) |
| `eigen_1d.in` | 最低四态 | \(E_n=n+1/2\)，Hermite 重叠 |
| `ground_1d_precise.in` | 更高精度基态 | \(\lvert\Delta E\rvert\sim10^{-7}\) |
| `eigen_1d_precise.in` | 更高精度四态 | \(\lvert\Delta E\rvert\sim10^{-5}\) |
| `ground_itp.in` | 虚时求基态 | \(E(\tau)\to 0.5\) |
| `ground_atom.in` | 一维软库仑基态 | 只看残差 |
| `orbitals_ground.in` | 两谐振子轨道，\(\lambda=0\) | \(\varepsilon=0.5,1.5\) |
| `helium_ground.in` | 精确一维双电子基态 | 只看残差 |
| `invert_smoke.in` | 一维双电子反演短跑 | \(n\) 的 L1 下降 |
| `invert_2e1d.in` | TGK08 一维氦反演 | 收回 \(v_\mathrm{ext}\) |
| `invert_2e2d.in` | 二维双电子当作 4 维单粒子 | 收回 \(v_\mathrm{ext}(x,y)\) |

演示输入网格较粗，为了尽快跑完。定态能量请用 `kinetic = 'bs'`，并把 `prec` 降到 `1d-5`–`1d-6`（见 `ground_1d_precise.in`）。

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
| 缺少奇宇称谐振子态 | 把 `x0` 设成非零，初猜不要取偶函数 |
| 残差降不下去 | 加大 `krylov_dim`、减小 `prec`、加大 `L`；能量请看重叠 / \(\Delta E\) |
| 重叠已经很高但能量对不上 | 平滑束缚态用 `kinetic = 'bs'`；收紧 `prec` 并不能改善 ABGV 的 \(\langle H\rangle\) |
| 虚时能量掉到 −∞ | 多小波动能没有下界；ITP 应走热核。若仍出现请确认二进制已重编 |
| 画不出图 | `pip install matplotlib`；在含 `.csv` / `.line` 的目录下运行 |

---

## 12. 限制

- Fetch 的 MRCPP 会打补丁以支持 `FunctionTree<4>`（Morton / 恒等 Hilbert 路径）。精确 N 体允许 \(D=\) `electrons × dim` \(\le 4\)：二维双电子，或四个一维电子，共用一棵树。更高维精确波函数请用 `mode = 'orbital'`。两电子反演（`calculation = 'invert'`）仍走均匀组态网格。
- 4 维树很重：每个节点 \((k+1)^4\) 个缩放系数、16 个子节点。请把 `order`、`prec`、`max_depth` 保持适中。\(D=4\) 时禁用虚时 `method = 'itp'`（4 维热核卷积）。
- 反演只在密度足够大的区域可信（论文经验：\(n\gtrsim 10^{-2}\)）。\(v_\mathrm{ext}\) 的加性常数通过在密度峰值处对齐来固定。
- `TimeEvolutionOperator` 仅 1D Legendre（传入 \(\tau=\Delta t/2\)，因为库里是 \(\exp(i\tau\partial_x^2)=\exp(-i T\Delta t)\)）。
- 接触 \(\lambda\rho\) 不是库仑 Hartree。
- MPI 不能把单棵树按空间分解。
- 定态 Lanczos 把与光滑初猜重叠大的 Ritz 向量当作物理低能态；多小波动能算子没有下界，因此不用代数上最小的 Ritz 值。
- 连续谱（`trap = 'free'`）得到的是盒子离散模，不是束缚态。

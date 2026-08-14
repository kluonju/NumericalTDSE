# NumericalTDSE

Adaptive multiwavelet solver of the time-dependent Schrödinger equation (TDSE), built on [MRCPP](https://github.com/MRChemSoft/mrcpp) (≥ 1.4, developed against the 1.6.0-alpha line that ships `TimeEvolutionOperator`).

The code represents wave functions as `FunctionTree` objects on a `MultiResolutionAnalysis`, refines the grid from local wavelet-norm error estimates down to a user-specified precision `prec`, and propagates `i ∂t ψ = H ψ` with one of three algorithms (Krylov / Strang split-operator / RK4).

---

## 物理与算法

原子单位下的 TDSE：

\[
i\,\partial_t\psi = \hat H(t)\,\psi,\qquad
\hat H = -\tfrac12\nabla^2 + V(\mathbf r,t).
\]

- **动能** \(T=-\frac12\nabla^2\)：默认用 MRCPP 推荐的 `ABGVOperator` 沿每个坐标方向作用两次；可选 `BSOperator` 或卷积形式的 `DerivativeConvolution`。
- **势能**：继承 `RepresentableFunction<D>`，每步 `project` 到 `FunctionTree`，再与波函数做 MW 乘法。含时激光取偶极近似 \(-E(t)\,x\)。
- **精确 N 体**（`mode = 'exact'`）：把 \(N\) 个 1D 电子写成一张 `FunctionTree<N>`（\(N\le 3\)，因为 MRCPP 显式实例化 D=1,2,3）。电子间软库仑 \(1/\sqrt{(x_i-x_j)^2+a^2}\)。
- **轨道平均场**（`mode = 'orbital'`）：最多 4 个轨道，每个是 `FunctionTree<dim>`，可选接触相互作用 \(\lambda\rho(\mathbf r)\)。四电子 3D 只能走这条路（12 维全波函数无法用 MW 表示）。

传播子：

| `propagator` | 公式 | 说明 |
|---|---|---|
| `krylov`（默认） | \(\psi\leftarrow\exp(-i\hat H\Delta t)\psi\) | 短迭代 Lanczos，任意维数，近幺正 |
| `split` | Strang：\(e^{-iV\Delta t/2}e^{-iT\Delta t}e^{-iV\Delta t/2}\) | 1D + Legendre 时动能指数是 `TimeEvolutionOperator` \(\exp(i(\Delta t/2)\partial_x^2)\)；更高维对 \(T\) 再用 Krylov |
| `rk4` | \(\partial_t\psi=-iH\psi\) 的四阶 Runge–Kutta | 实现最直观，非幺正，步长宜小 |

`IdentityConvolution` 在 \(t=0\) 作用一次，检查 \(\|I\psi-\psi\|\)（`ident_check = .false.` 关掉）。

---

## 依赖与编译

- C++17 编译器，CMake ≥ 3.16
- [MRCPP](https://github.com/MRChemSoft/mrcpp)（CMake 在找不到本地安装时会 FetchContent 指定 commit）
- Eigen3（由 MRCPP 自动拉取）
- 可选 OpenMP

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNUMTDSE_OPENMP=ON -DCMAKE_CXX_COMPILER=g++
cmake --build build -j --target tdse
ctest --test-dir build --output-on-failure
```

若已安装 MRCPP：

```bash
cmake -S . -B build -DMRCPP_DIR=$HOME/Software/mrcpp/share/cmake/MRCPP
```

OpenMP 线程数：`export OMP_NUM_THREADS=8`。

---

## 运行

输入是 Quantum ESPRESSO 风格的自由格式 namelist：只写需要改的关键字，其余保持默认。注释用 `!` 或 `#`，实数可用 Fortran `1.0d-4`，逻辑值为 `.true.` / `.false.`。

```bash
./build/bin/tdse examples/harmonic_1d.in
./build/bin/tdse -i examples/free_particle.in
./build/bin/tdse --template          # 打印带注释的完整模板
./build/bin/tdse --smoke             # 内置短跑，供 ctest
```

段名：`&CONTROL` `&MRA` `&TIME` `&SYSTEM` `&INITIAL` `&LASER` `&OUTPUT`  
别名：`CTRL`；`GRID` / `NUMERICS` → `MRA`；`PROPAGATOR` → `TIME`；`WAVEFUNCTION` / `PSI` → `INITIAL`；`FIELD` → `LASER`；`IO` / `OUT` → `OUTPUT`。

```fortran
! 最小输入：1D 谐振子，其余全是默认值
&SYSTEM
  trap = 'harmonic'
/

&INITIAL
  x0 = 1.0
/
```

`examples/` 里还有 `free_particle.in`、`helium_1d.in`、`orbitals_4e.in`、`split_1d.in`、`smoke.in`。`calculation = 'smoke'` 会先载入短跑预设，文件里显式写出的关键字仍然生效。

CSV 列：`t, norm, dipole, energy, nodes_re, nodes_im, overlap_analytic`。`prefix = 'job'` 且未写 `output=` 时，观测文件为 `job_observables.csv`。

---

## 代码结构

```
include/tdse/
  parameters.hpp      精度 / 盒子 / 时间步 / 物理模型
  input.hpp           QE 风格 namelist 解析
  analytic.hpp        RepresentableFunction：初态与含时势能
  wavefunction.hpp    复数波函数 = (Re FunctionTree, Im FunctionTree)
  operators.hpp       MRA 算子：ABGV、IdentityConvolution、动能 / 势能作用
  propagator.hpp      Split / Krylov / RK4
  observables.hpp     模方、偶极、能量
  simulate.hpp        时间循环
src/main.cpp          入口
src/parameters.cpp    CLI（input.in / --template / --smoke）
src/input.cpp         namelist 词法与赋值
```

可调参数集中在 `Parameters`：`prec`, `order`, `max_depth`, `L`, `dt`, `T`。完整关键字见 `tdse --template`。

---

## 约定与限制

- 哈密顿量使用标准量子化学原子单位 \(T=-\frac12\nabla^2\)。MRCPP 的 `TimeEvolutionOperator` 表示 \(\exp(i\tau\partial_x^2)\)，因此传入 \(\tau=\Delta t/2\)。
- 该算子目前只实现于 **1D Legendre** 缩放函数。
- MRCPP 对 `FunctionTree<D>` 显式实例化 D=1,2,3。四电子精确波函数请用 `mode = 'orbital'`。
- 默认演示参数偏向“能跑完”，要定量结果请把 `prec` 降到 `1d-5`–`1d-6` 并减小 `dt`。

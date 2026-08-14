#!/usr/bin/env python3
"""Patch a fetched MRCPP source tree so FunctionTree<4> can be instantiated.

Upstream MRCPP explicitly instantiates D = 1, 2, 3 only. NumericalTDSE needs
D = 4 for two electrons in 2D configuration space. This script is idempotent
and is meant to run as a CMake PATCH_COMMAND / execute_process on the
FetchContent source directory.

Algorithmic changes:
    * HilbertPath<4> is a header-only identity/Morton specialization (no 16-child
      Hilbert tables are shipped).
    * tree_utils::mw_transform gains a fourth filter pass.
    * Quadrature / tensor coordinate expansion gain a 4D branch.
    * OperatorStatistics component counters grow from 8x8 (2^3) to 16x16 (2^4).
    * NodeIndex::operator< compares all D translation indices (upstream stops at 3).
    * Explicit instantiations of D = 3 templates are cloned to D = 4, skipping
      D = 3-only specializations (mw_transform_back, CompFunction, Poisson, ...).
"""

from __future__ import annotations

import pathlib
import re
import sys

MARKER = "NUMTDSE_MRCPP_D4"

SKIP_FILES = {
    "CompFunction.cpp",
    "HilbertPath.cpp",
    "parallel.cpp",
    "CartesianConvolution.cpp",
    "PoissonOperator.cpp",
    "HelmholtzOperator.cpp",
}

SKIP_SUBSTRINGS = (
    "mw_transform_back",
    "HilbertPath",
    "CompFunction",
    "metric",
    "saveNodesAndRmCoeff",
    "reCompress",
    "integrateEndNodes",
    "Poisson",
    "Helmholtz",
    "CartesianConvolution",
    "broadcast_Tree_noCoeff",
    "reduce_Tree_noCoeff",
    "allreduce_Tree_noCoeff",
)

HILBERT4 = r"""
/** Identity / Morton child numbering for D = 4.
 *  Upstream Hilbert tables are sized [][8] = 2^3 children. D = 4 needs 16.
 *  Morton order is enough for MW correctness; spatial locality is slightly
 *  worse than a true Hilbert curve.
 *  """ + MARKER + r""" */
template <> class HilbertPath<4> final {
public:
    HilbertPath() = default;
    HilbertPath(const HilbertPath<4> &p)
            : path(p.path) {}
    HilbertPath(const HilbertPath<4> & /*p*/, int /*cIdx*/)
            : path(0) {}
    HilbertPath &operator=(const HilbertPath<4> &p) {
        this->path = p.path;
        return *this;
    }

    short int getPath() const { return this->path; }
    short int getChildPath(int /*hIdx*/) const { return 0; }
    int getZIndex(int hIdx) const { return hIdx; }
    int getHIndex(int zIdx) const { return zIdx; }

private:
    short int path{0};
};

"""

TENSOR4_H = (
    "void tensor_expand_coords_4D(int kp1, const Eigen::MatrixXd &primitive, "
    "Eigen::MatrixXd &expanded);\n"
)

TENSOR4_CPP = r"""
void math_utils::tensor_expand_coords_4D(int kp1, const MatrixXd &primitive, MatrixXd &expanded) {
    int n = 0;
    for (int i = 0; i < kp1; i++) {
        for (int j = 0; j < kp1; j++) {
            for (int k = 0; k < kp1; k++) {
                for (int l = 0; l < kp1; l++) {
                    expanded(0, n) = primitive(0, l);
                    expanded(1, n) = primitive(1, k);
                    expanded(2, n) = primitive(2, j);
                    expanded(3, n) = primitive(3, i);
                    n++;
                }
            }
        }
    }
}

"""

MW_TRANSFORM_D4 = r"""
    if (D == 4) {
        // Direction 2: tmpcoeff2 -> tmpcoeff3 (intermediate, like dir 1).
        i++;
        mask = 4; // 1 << i;
        for (int gt = 0; gt < tDim; gt++) {
            T *out = tmpcoeff3.data() + gt * kp1_d;
            for (int ft = 0; ft < ftlim3; ft++) {
                if ((gt | mask) == (ft | mask)) {
                    T *in = tmpcoeff2.data() + ft * kp1_d;
                    int filter_index = 2 * ((gt >> i) & 1) + ((ft >> i) & 1);
                    const Eigen::MatrixXd &oper = filter.getSubFilter(filter_index, operation);

                    math_utils::apply_filter(out, in, oper, kp1, kp1_dm1, overwrite);
                    overwrite = 1.0;
                }
            }
            overwrite = 0.0;
        }
        // Direction 3: tmpcoeff3 -> coeff_out (last dir, same overwrite/stride as D=3).
        overwrite = 1.0;
        if (b_overwrite) overwrite = 0.0;
        i++;
        mask = 8; // 1 << i;
        for (int gt = 0; gt < tDim; gt++) {
            T *out = coeff_out + gt * stride;
            for (int ft = 0; ft < ftlim4; ft++) {
                if ((gt | mask) == (ft | mask)) {
                    T *in = tmpcoeff3.data() + ft * kp1_d;
                    int filter_index = 2 * ((gt >> i) & 1) + ((ft >> i) & 1);
                    const Eigen::MatrixXd &oper = filter.getSubFilter(filter_index, operation);

                    math_utils::apply_filter(out, in, oper, kp1, kp1_dm1, overwrite);
                    overwrite = 1.0;
                }
            }
            overwrite = 1.0;
            if (b_overwrite) overwrite = 0.0;
        }
    }

    if (D > 4) MSG_ABORT("D>4 NOT IMPLEMENTED for S_mwtransform");
"""

INTEGRATE4 = r"""    if (D > 4)
        MSG_ABORT("Not Implemented")
    else if (D == 4) {
        for (int i = 0; i < qOrder; i++) {
            T sumj = 0.0;
            for (int j = 0; j < qOrder; j++) {
                T sumk = 0.0;
                for (int k = 0; k < qOrder; k++) {
                    T suml = 0.0;
                    for (int l = 0; l < qOrder; l++) suml += cc[nc++] * weights[l];
                    sumk += suml * weights[k];
                }
                sumj += sumk * weights[j];
            }
            sum += sumj * weights[i];
        }
    } else if (D == 3) {
"""


def die(msg: str) -> None:
    print(f"patch_mrcpp_d4: {msg}", file=sys.stderr)
    sys.exit(1)


def replace_once(text: str, old: str, new: str, path: pathlib.Path) -> str:
    if old not in text:
        die(f"{path}: expected snippet not found:\n{old[:120]}")
    return text.replace(old, new, 1)


def dim3_to_dim4(stmt: str) -> str:
    out = stmt
    out = out.replace("<3,", "<4,")
    out = out.replace("<3>", "<4>")
    out = out.replace("<3>(", "<4>(")
    out = out.replace(", 3>", ", 4>")
    out = out.replace(", 3>(", ", 4>(")
    out = out.replace("array<double, 3>", "array<double, 4>")
    out = out.replace("array<int, 3>", "array<int, 4>")
    out = out.replace("array<bool, 3>", "array<bool, 4>")
    return out


def should_clone(stmt: str) -> bool:
    if "<3" not in stmt and ", 3>" not in stmt and "array<" not in stmt:
        return False
    if "<3" not in stmt and ", 3>" not in stmt:
        return False
    for needle in SKIP_SUBSTRINGS:
        if needle in stmt:
            return False
    return True


def clone_instantiations(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        is_explicit = (
            stripped.startswith("template ")
            and not stripped.startswith("template <")
            and not stripped.startswith("template<>")
            and not stripped.startswith("template <>")
        )
        if is_explicit:
            block = [lines[i]]
            while ";" not in "".join(block):
                i += 1
                if i >= len(lines):
                    die("unterminated template instantiation")
                block.append(lines[i])
            stmt = "".join(block)
            out.append(stmt)
            cloned = dim3_to_dim4(stmt)
            if cloned != stmt and should_clone(stmt) and cloned not in text and cloned not in "".join(out):
                out.append(cloned)
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return "".join(out)


def patch_hilbert(root: pathlib.Path) -> None:
    path = root / "src/trees/HilbertPath.h"
    text = path.read_text()
    if "HilbertPath<4>" in text:
        return
    needle = "} // namespace mrcpp"
    idx = text.rfind(needle)
    if idx < 0:
        die(f"{path}: closing namespace not found")
    path.write_text(text[:idx] + HILBERT4 + text[idx:])


def patch_math_utils(root: pathlib.Path) -> None:
    header = root / "src/utils/math_utils.h"
    h = header.read_text()
    if "tensor_expand_coords_4D" not in h:
        old = "void tensor_expand_coords_3D(int kp1, const Eigen::MatrixXd &primitive, Eigen::MatrixXd &expanded);\n"
        if old not in h:
            die(f"{header}: 3D tensor expand declaration missing")
        header.write_text(h.replace(old, old + TENSOR4_H, 1))

    src = root / "src/utils/math_utils.cpp"
    c = src.read_text()
    if "tensor_expand_coords_4D" not in c:
        # Insert after the 3D expander, before the hermitian_matrix_pow comment.
        old = """                expanded(2, n) = primitive(2, i);
                n++;
            }
        }
    }
}

/** @brief Compute the eigenvalues and eigenvectors of a Hermitian matrix
"""
        new = """                expanded(2, n) = primitive(2, i);
                n++;
            }
        }
    }
}
""" + TENSOR4_CPP + """/** @brief Compute the eigenvalues and eigenvectors of a Hermitian matrix
"""
        c = replace_once(c, old, new, src)
    if "calc_distance<4>" not in c:
        old = "template double math_utils::calc_distance<3>(const Coord<3> &a, const Coord<3> &b);\n"
        new = old + "template double math_utils::calc_distance<4>(const Coord<4> &a, const Coord<4> &b);\n"
        c = replace_once(c, old, new, src)
    src.write_text(c)


def patch_mw_transform(root: pathlib.Path) -> None:
    path = root / "src/utils/tree_utils.cpp"
    text = path.read_text()
    if "ftlim4" in text and "tmpcoeff3" in text:
        return

    text = replace_once(
        text,
        "    std::vector<T> tmpcoeff(kp1_d * tDim);\n"
        "    std::vector<T> tmpcoeff2(kp1_d * tDim);\n"
        "    int ftlim = tDim;\n"
        "    int ftlim2 = tDim;\n"
        "    int ftlim3 = tDim;\n"
        "    if (readOnlyScaling) {\n"
        "        ftlim = 1;\n"
        "        ftlim2 = 2;\n"
        "        ftlim3 = 4;\n",
        "    std::vector<T> tmpcoeff(kp1_d * tDim);\n"
        "    std::vector<T> tmpcoeff2(kp1_d * tDim);\n"
        "    std::vector<T> tmpcoeff3(kp1_d * tDim);\n"
        "    int ftlim = tDim;\n"
        "    int ftlim2 = tDim;\n"
        "    int ftlim3 = tDim;\n"
        "    int ftlim4 = tDim;\n"
        "    if (readOnlyScaling) {\n"
        "        ftlim = 1;\n"
        "        ftlim2 = 2;\n"
        "        ftlim3 = 4;\n"
        "        ftlim4 = 8;\n",
        path,
    )
    text = replace_once(text, "    if (D > 2) {", "    if (D == 3) {", path)
    text = replace_once(
        text,
        '    if (D > 3) MSG_ABORT("D>3 NOT IMPLEMENTED for S_mwtransform");\n',
        MW_TRANSFORM_D4 + "\n",
        path,
    )
    path.write_text(text)


def patch_mwnode(root: pathlib.Path) -> None:
    path = root / "src/trees/MWNode.cpp"
    text = path.read_text()
    if "tensor_expand_coords_4D" in text:
        return
    text = text.replace(
        "    if (D == 3) math_utils::tensor_expand_coords_3D(kp1, prim_pts, pts);\n"
        "    if (D >= 4) NOT_IMPLEMENTED_ABORT;",
        "    if (D == 3) math_utils::tensor_expand_coords_3D(kp1, prim_pts, pts);\n"
        "    if (D == 4) math_utils::tensor_expand_coords_4D(kp1, prim_pts, pts);\n"
        "    if (D >= 5) NOT_IMPLEMENTED_ABORT;",
    )
    text = text.replace(
        "        if (D == 3) math_utils::tensor_expand_coords_3D(kp1, prim_t, exp_t);\n"
        "        if (D >= 4) NOT_IMPLEMENTED_ABORT;",
        "        if (D == 3) math_utils::tensor_expand_coords_3D(kp1, prim_t, exp_t);\n"
        "        if (D == 4) math_utils::tensor_expand_coords_4D(kp1, prim_t, exp_t);\n"
        "        if (D >= 5) NOT_IMPLEMENTED_ABORT;",
    )
    if "tensor_expand_coords_4D" not in text:
        die(f"{path}: failed to insert 4D tensor expand")
    path.write_text(text)


def patch_function_node(root: pathlib.Path) -> None:
    path = root / "src/trees/FunctionNode.cpp"
    text = path.read_text()
    if "else if (D == 4)" in text and "suml" in text:
        return
    old = """    if (D > 3)
        MSG_ABORT("Not Implemented")
    else if (D == 3) {
"""
    text = replace_once(text, old, INTEGRATE4, path)
    path.write_text(text)


def patch_operator_statistics(root: pathlib.Path) -> None:
    """D = 4 has 16 scaling/wavelet components; upstream counters are 8x8."""
    for rel in (
        "src/operators/OperatorStatistics.h",
        "src/operators/OperatorStatistics.cpp",
    ):
        path = root / rel
        text = path.read_text()
        if "Matrix<int, 16, 16>" in text:
            continue
        if "Matrix<int, 8, 8>" not in text:
            die(f"{path}: expected 8x8 component-count matrix")
        path.write_text(text.replace("Matrix<int, 8, 8>", "Matrix<int, 16, 16>"))


def patch_node_index(root: pathlib.Path) -> None:
    path = root / "src/trees/NodeIndex.h"
    text = path.read_text()
    if "for (int d = 0; d < D; d++)" in text and "idx.L[d] != idy.L[d]" in text:
        return
    old = """    bool operator<(const NodeIndex<D> &idy) const {
        const NodeIndex<D> &idx = *this;
        if (idx.N != idy.N) return idx.N < idy.N;
        if (idx.L[0] != idy.L[0] or D < 2) return idx.L[0] < idy.L[0];
        if (idx.L[1] != idy.L[1] or D < 3) return idx.L[1] < idy.L[1];
        return idx.L[2] < idy.L[2];
    }
"""
    new = """    bool operator<(const NodeIndex<D> &idy) const {
        const NodeIndex<D> &idx = *this;
        if (idx.N != idy.N) return idx.N < idy.N;
        for (int d = 0; d < D; d++) {
            if (idx.L[d] != idy.L[d]) return idx.L[d] < idy.L[d];
        }
        return false;
    }
"""
    path.write_text(replace_once(text, old, new, path))


def walk_and_instantiate(root: pathlib.Path) -> int:
    src = root / "src"
    n_files = 0
    for path in sorted(src.rglob("*.cpp")):
        if path.name in SKIP_FILES:
            continue
        rel = path.relative_to(root)
        if rel.parts[0] == "tests" or "tests" in rel.parts:
            continue
        original = path.read_text()
        updated = clone_instantiations(original)
        if updated != original:
            path.write_text(updated)
            n_files += 1
    return n_files


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    header = root / "src/trees/HilbertPath.h"
    if not header.is_file():
        die(f"not an MRCPP source tree: {root}")

    patch_hilbert(root)
    patch_math_utils(root)
    patch_mw_transform(root)
    patch_mwnode(root)
    patch_function_node(root)
    patch_operator_statistics(root)
    patch_node_index(root)
    n = walk_and_instantiate(root)
    stamp = root / ".numtdse_mrcpp_d4"
    stamp.write_text(MARKER + "\n")
    print(f"patch_mrcpp_d4: patched {root} ({n} instantiation files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

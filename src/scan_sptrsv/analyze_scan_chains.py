#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CSV_FIELDS = [
    "matrix_name",
    "n_rows",
    "nnz",
    "absolute_head_rows",
    "absolute_head_row_ratio",
    "strict_boundary_head_rows",
    "strict_boundary_head_row_ratio",
    "strict_single_pred_rows",
    "strict_single_pred_row_ratio",
    "strict_multi_pred_rows",
    "strict_multi_pred_row_ratio",
    "strict_branch_rows",
    "strict_branch_row_ratio",
    "strict_chain_count",
    "strict_chain_row_coverage",
    "strict_chain_row_coverage_ratio",
    "max_strict_chain_length",
    "avg_strict_chain_length",
    "strict_chain_len_ge_2",
    "strict_chain_len_ge_4",
    "strict_chain_len_ge_8",
    "strict_chain_len_ge_16",
    "strict_chain_len_ge_32",
    "strict_chain_len_ge_64",
]

FRONTIER_SAFE_CSV_FIELDS = [
    "frontier_safe_max_level",
    "frontier_safe_eligible_rows",
    "frontier_safe_eligible_row_ratio",
    "frontier_safe_invalid_multi_deepest_rows",
    "frontier_safe_invalid_multi_deepest_row_ratio",
    "frontier_safe_branch_rows",
    "frontier_safe_branch_row_ratio",
    "frontier_safe_boundary_cut_rows",
    "frontier_safe_chain_count",
    "frontier_safe_chain_row_coverage",
    "frontier_safe_chain_row_coverage_ratio",
    "max_frontier_safe_chain_length",
    "avg_frontier_safe_chain_length",
    "frontier_safe_chain_len_ge_2",
    "frontier_safe_chain_len_ge_4",
    "frontier_safe_chain_len_ge_8",
    "frontier_safe_chain_len_ge_16",
    "frontier_safe_chain_len_ge_32",
    "frontier_safe_chain_len_ge_64",
]

DEFAULT_MAX_DUMP_ROWS = 1_000_000
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"
MATRIX_DIR = DATA_DIR / "matrices"
REAL_MATRIX_DIR = MATRIX_DIR / "real"
REGRESSION_MATRIX_DIR = MATRIX_DIR / "regression"
SPCG_FACTOR_DIR = DATA_DIR / "factors" / "spcg"
RESULTS_DIR = REPO_ROOT / "results"
STRICT_RESULTS_DIR = RESULTS_DIR / "strict"
FRONTIER_SAFE_RESULTS_DIR = RESULTS_DIR / "frontier_safe"
REGRESSION_RESULTS_DIR = RESULTS_DIR / "regression"
DEFAULT_MATRIX_LOWER_OUT = STRICT_RESULTS_DIR / "matrix_lower.csv"
DEFAULT_SPCG_FACTOR_OUT = STRICT_RESULTS_DIR / "spcg_factor_l.csv"
DEFAULT_FRONTIER_SAFE_MATRIX_LOWER_OUT = FRONTIER_SAFE_RESULTS_DIR / "matrix_lower.csv"
DEFAULT_FRONTIER_SAFE_SPCG_FACTOR_OUT = FRONTIER_SAFE_RESULTS_DIR / "spcg_factor_l.csv"
DEFAULT_SMALL_TEST_INPUT = REGRESSION_MATRIX_DIR
DEFAULT_SMALL_TEST_OUT = REGRESSION_RESULTS_DIR / "strict_small_latest.csv"
DEFAULT_SMALL_TEST_FRONTIER_OUT = REGRESSION_RESULTS_DIR / "frontier_safe_small_latest.csv"
DEFAULT_FACTOR_DIR = SPCG_FACTOR_DIR

CSV_FIELD_LABELS = {
    "matrix_name": "矩阵名",
    "n_rows": "行数",
    "nnz": "非零元数",
    "absolute_head_rows": "绝对链头行数",
    "absolute_head_row_ratio": "绝对链头行占比",
    "strict_boundary_head_rows": "严格边界链头行数",
    "strict_boundary_head_row_ratio": "严格边界链头行占比",
    "strict_single_pred_rows": "严格单前驱行数",
    "strict_single_pred_row_ratio": "严格单前驱行占比",
    "strict_multi_pred_rows": "多前驱行数",
    "strict_multi_pred_row_ratio": "多前驱行占比",
    "strict_branch_rows": "严格分叉节点数",
    "strict_branch_row_ratio": "严格分叉节点占比",
    "strict_chain_count": "严格链条数",
    "strict_chain_row_coverage": "严格链覆盖节点数",
    "strict_chain_row_coverage_ratio": "严格链覆盖占比",
    "max_strict_chain_length": "最长严格链长度",
    "avg_strict_chain_length": "平均严格链长度",
    "strict_chain_len_ge_2": "长度至少2的严格链条数",
    "strict_chain_len_ge_4": "长度至少4的严格链条数",
    "strict_chain_len_ge_8": "长度至少8的严格链条数",
    "strict_chain_len_ge_16": "长度至少16的严格链条数",
    "strict_chain_len_ge_32": "长度至少32的严格链条数",
    "strict_chain_len_ge_64": "长度至少64的严格链条数",
    "frontier_safe_max_level": "frontier-safe 最大依赖层级",
    "frontier_safe_eligible_rows": "frontier-safe 可扫描行数",
    "frontier_safe_eligible_row_ratio": "frontier-safe 可扫描行占比",
    "frontier_safe_invalid_multi_deepest_rows": "并列最深前驱冲突行数",
    "frontier_safe_invalid_multi_deepest_row_ratio": "并列最深前驱冲突行占比",
    "frontier_safe_branch_rows": "frontier-safe 分叉节点数",
    "frontier_safe_branch_row_ratio": "frontier-safe 分叉节点占比",
    "frontier_safe_boundary_cut_rows": "frontier-safe 边界切链次数",
    "frontier_safe_chain_count": "frontier-safe 链条数",
    "frontier_safe_chain_row_coverage": "frontier-safe 链覆盖节点数",
    "frontier_safe_chain_row_coverage_ratio": "frontier-safe 链覆盖占比",
    "max_frontier_safe_chain_length": "最长 frontier-safe 链长度",
    "avg_frontier_safe_chain_length": "平均 frontier-safe 链长度",
    "frontier_safe_chain_len_ge_2": "长度至少2的 frontier-safe 链条数",
    "frontier_safe_chain_len_ge_4": "长度至少4的 frontier-safe 链条数",
    "frontier_safe_chain_len_ge_8": "长度至少8的 frontier-safe 链条数",
    "frontier_safe_chain_len_ge_16": "长度至少16的 frontier-safe 链条数",
    "frontier_safe_chain_len_ge_32": "长度至少32的 frontier-safe 链条数",
    "frontier_safe_chain_len_ge_64": "长度至少64的 frontier-safe 链条数",
}


@dataclass(frozen=True)
class MatrixMarketHeader:
    n_rows: int
    n_cols: int
    reported_nnz: int
    field: str
    symmetry: str

    @property
    def expands_symmetric_structure(self):
        return self.symmetry in {"symmetric", "hermitian", "skew-symmetric"}


@dataclass(frozen=True)
class StrictLowerDependencies:
    n_rows: int
    strict_lower_nnz: int
    pred_count: np.ndarray
    first_pred: np.ndarray


@dataclass(frozen=True)
class StrictLowerDependencyGraph:
    n_rows: int
    strict_lower_nnz: int
    pred_count: np.ndarray
    first_pred: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray


@dataclass(frozen=True)
class StrictChildArrays:
    strict_child_count: np.ndarray
    unique_strict_child: np.ndarray


def _index_dtype(n_rows: int):
    if n_rows <= np.iinfo(np.int32).max:
        return np.int32
    return np.int64


def _count_dtype(max_count: int):
    if max_count <= np.iinfo(np.int32).max:
        return np.int32
    return np.int64


def _parse_matrix_market_banner(line: str, path: Path):
    tokens = line.strip().split()
    if len(tokens) != 5 or tokens[0] != "%%MatrixMarket":
        raise ValueError(f"Invalid Matrix Market banner in {path.name}")

    object_type = tokens[1].lower()
    storage_format = tokens[2].lower()
    field = tokens[3].lower()
    symmetry = tokens[4].lower()

    if object_type != "matrix" or storage_format != "coordinate":
        raise ValueError(
            f"Only Matrix Market coordinate matrices are supported: {path.name}"
        )

    if field not in {"real", "integer", "pattern", "complex"}:
        raise ValueError(f"Unsupported Matrix Market field '{field}' in {path.name}")

    if symmetry not in {"general", "symmetric", "hermitian", "skew-symmetric"}:
        raise ValueError(
            f"Unsupported Matrix Market symmetry '{symmetry}' in {path.name}"
        )

    return field, symmetry


def _read_matrix_market_size(file_obj, path: Path, start_line_number: int):
    line_number = start_line_number
    for line in file_obj:
        line_number += 1
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue

        parts = stripped.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid Matrix Market size line in {path.name}")

        try:
            n_rows = int(parts[0])
            n_cols = int(parts[1])
            reported_nnz = int(parts[2])
        except ValueError as exc:
            raise ValueError(f"Invalid Matrix Market size line in {path.name}") from exc

        return n_rows, n_cols, reported_nnz, line_number

    raise ValueError(f"Missing Matrix Market size line in {path.name}")


def read_strict_lower_dependencies_mtx(path: Path, triangle: str = "lower"):
    """
    Stream a Matrix Market file and extract only strict lower dependencies.

    Output arrays are flat NumPy arrays:
    - pred_count[i] counts lower-triangular predecessors of row i
    - first_pred[i] stores the first predecessor seen for row i, or -1
    """
    if triangle != "lower":
        raise ValueError(f"Unsupported triangle type: {triangle}")

    with path.open("r", encoding="utf-8", errors="replace") as f:
        banner = f.readline()
        if not banner:
            raise ValueError(f"Empty Matrix Market file: {path.name}")

        field, symmetry = _parse_matrix_market_banner(banner, path)
        n_rows, n_cols, reported_nnz, line_number = _read_matrix_market_size(
            f, path, start_line_number=1
        )

        header = MatrixMarketHeader(
            n_rows=n_rows,
            n_cols=n_cols,
            reported_nnz=reported_nnz,
            field=field,
            symmetry=symmetry,
        )
        if header.n_rows != header.n_cols:
            raise ValueError(
                f"Matrix is not square: {path.name}, shape=({n_rows}, {n_cols})"
            )

        index_dtype = _index_dtype(header.n_rows)
        count_dtype = _count_dtype(header.reported_nnz)
        pred_count = np.zeros(header.n_rows, dtype=count_dtype)
        first_pred = np.full(header.n_rows, -1, dtype=index_dtype)

        strict_lower_nnz = 0
        entries_read = 0
        expands_symmetric = header.expands_symmetric_structure

        for line in f:
            line_number += 1
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue

            parts = stripped.split(maxsplit=2)
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid Matrix Market entry at {path.name}:{line_number}"
                )

            entries_read += 1
            try:
                row = int(parts[0]) - 1
                col = int(parts[1]) - 1
            except ValueError as exc:
                raise ValueError(
                    f"Invalid Matrix Market indices at {path.name}:{line_number}"
                ) from exc

            if row < 0 or row >= header.n_rows or col < 0 or col >= header.n_cols:
                raise ValueError(
                    f"Matrix Market index out of bounds at {path.name}:{line_number}"
                )

            if row > col:
                pred_count[row] += 1
                if first_pred[row] == -1:
                    first_pred[row] = col
                strict_lower_nnz += 1
            elif expands_symmetric and col > row:
                pred_count[col] += 1
                if first_pred[col] == -1:
                    first_pred[col] = row
                strict_lower_nnz += 1

        if entries_read != header.reported_nnz:
            raise ValueError(
                f"{path.name} reported {header.reported_nnz} entries but "
                f"streamed {entries_read}"
            )

    return StrictLowerDependencies(
        n_rows=header.n_rows,
        strict_lower_nnz=strict_lower_nnz,
        pred_count=pred_count,
        first_pred=first_pred,
    )


def read_strict_lower_dependency_graph_mtx(path: Path, triangle: str = "lower"):
    """
    Stream a Matrix Market file into a compact strict-lower dependency graph.

    This is used only for analyses that need predecessor levels. The default
    strict-chain path keeps the lighter count-only reader above.
    """
    deps = read_strict_lower_dependencies_mtx(path, triangle=triangle)
    index_dtype = _index_dtype(deps.n_rows)
    ptr_dtype = _count_dtype(deps.strict_lower_nnz)

    indptr = np.zeros(deps.n_rows + 1, dtype=ptr_dtype)
    indptr[1:] = np.cumsum(deps.pred_count, dtype=ptr_dtype)
    indices = np.empty(deps.strict_lower_nnz, dtype=index_dtype)
    cursor = indptr[:-1].copy()

    with path.open("r", encoding="utf-8", errors="replace") as f:
        banner = f.readline()
        if not banner:
            raise ValueError(f"Empty Matrix Market file: {path.name}")

        field, symmetry = _parse_matrix_market_banner(banner, path)
        n_rows, n_cols, reported_nnz, line_number = _read_matrix_market_size(
            f, path, start_line_number=1
        )
        header = MatrixMarketHeader(
            n_rows=n_rows,
            n_cols=n_cols,
            reported_nnz=reported_nnz,
            field=field,
            symmetry=symmetry,
        )
        expands_symmetric = header.expands_symmetric_structure

        for line in f:
            line_number += 1
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue

            parts = stripped.split(maxsplit=2)
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid Matrix Market entry at {path.name}:{line_number}"
                )

            try:
                row = int(parts[0]) - 1
                col = int(parts[1]) - 1
            except ValueError as exc:
                raise ValueError(
                    f"Invalid Matrix Market indices at {path.name}:{line_number}"
                ) from exc

            if row > col:
                write_at = int(cursor[row])
                indices[write_at] = col
                cursor[row] += 1
            elif expands_symmetric and col > row:
                write_at = int(cursor[col])
                indices[write_at] = row
                cursor[col] += 1

    if not np.array_equal(cursor, indptr[1:]):
        raise RuntimeError(f"Failed to fill strict lower dependency graph: {path.name}")

    return StrictLowerDependencyGraph(
        n_rows=deps.n_rows,
        strict_lower_nnz=deps.strict_lower_nnz,
        pred_count=deps.pred_count,
        first_pred=deps.first_pred,
        indptr=indptr,
        indices=indices,
    )


def build_strict_parent(pred_count, first_pred):
    """Reuse first_pred storage as strict_parent for exact single-predecessor rows."""
    strict_parent = first_pred
    strict_parent[pred_count != 1] = -1
    return strict_parent


def build_strict_child_arrays(strict_parent):
    """Build strict child counts without Python list-of-lists reverse adjacency."""
    n = len(strict_parent)
    strict_child_count = np.zeros(n, dtype=np.int32)
    unique_strict_child = np.full(n, -1, dtype=strict_parent.dtype)

    for child, parent in enumerate(strict_parent):
        parent = int(parent)
        if parent == -1:
            continue

        child_count = int(strict_child_count[parent])
        if child_count == 0:
            unique_strict_child[parent] = child
        elif child_count == 1:
            unique_strict_child[parent] = -1
        strict_child_count[parent] = child_count + 1

    return StrictChildArrays(
        strict_child_count=strict_child_count,
        unique_strict_child=unique_strict_child,
    )


def _is_strict_boundary_head(i, pred_count, strict_parent, strict_child_count):
    if strict_child_count[i] == 0:
        return False

    if pred_count[i] != 1:
        return True

    parent = int(strict_parent[i])
    if parent == -1:
        return True

    return strict_child_count[parent] != 1


def analyze_strict_chains_from_arrays(
    pred_count,
    strict_parent,
    strict_child_count,
    unique_strict_child,
    dump_chains=False,
):
    """Compute strict-chain stats by walking flat arrays only."""
    n_rows = len(pred_count)

    absolute_head_rows = int(np.count_nonzero(pred_count == 0))
    strict_single_pred_rows = int(np.count_nonzero(pred_count == 1))
    strict_multi_pred_rows = int(np.count_nonzero(pred_count > 1))
    strict_branch_rows = int(np.count_nonzero(strict_child_count > 1))

    strict_boundary_head_rows = 0
    strict_chain_count = 0
    strict_chain_row_coverage = 0
    max_strict_chain_length = 0
    strict_chain_len_ge_2 = 0
    strict_chain_len_ge_4 = 0
    strict_chain_len_ge_8 = 0
    strict_chain_len_ge_16 = 0
    strict_chain_len_ge_32 = 0
    strict_chain_len_ge_64 = 0
    dumped = [] if dump_chains else None

    for head in range(n_rows):
        if not _is_strict_boundary_head(
            head, pred_count, strict_parent, strict_child_count
        ):
            continue

        strict_boundary_head_rows += 1
        length = 1
        current = head
        nodes = [head] if dump_chains else None

        while strict_child_count[current] == 1:
            child = int(unique_strict_child[current])
            if child == -1:
                raise RuntimeError(f"Missing unique strict child for row {current}")
            current = child
            length += 1
            if dump_chains:
                nodes.append(current)

        if length < 2:
            continue

        strict_chain_count += 1
        strict_chain_row_coverage += length
        max_strict_chain_length = max(max_strict_chain_length, length)

        if length >= 2:
            strict_chain_len_ge_2 += 1
        if length >= 4:
            strict_chain_len_ge_4 += 1
        if length >= 8:
            strict_chain_len_ge_8 += 1
        if length >= 16:
            strict_chain_len_ge_16 += 1
        if length >= 32:
            strict_chain_len_ge_32 += 1
        if length >= 64:
            strict_chain_len_ge_64 += 1

        if dump_chains:
            dumped.append(
                {
                    "head": head,
                    "head_type": "absolute_head"
                    if pred_count[head] == 0
                    else "boundary_head",
                    "head_pred_count": int(pred_count[head]),
                    "length": length,
                    "nodes": nodes,
                }
            )

    avg_strict_chain_length = (
        strict_chain_row_coverage / strict_chain_count
        if strict_chain_count > 0
        else 0.0
    )

    stats = {
        "absolute_head_rows": absolute_head_rows,
        "absolute_head_row_ratio": (absolute_head_rows / n_rows) if n_rows > 0 else 0.0,
        "strict_boundary_head_rows": strict_boundary_head_rows,
        "strict_boundary_head_row_ratio": (
            strict_boundary_head_rows / n_rows
        ) if n_rows > 0 else 0.0,
        "strict_single_pred_rows": strict_single_pred_rows,
        "strict_single_pred_row_ratio": (
            strict_single_pred_rows / n_rows
        ) if n_rows > 0 else 0.0,
        "strict_multi_pred_rows": strict_multi_pred_rows,
        "strict_multi_pred_row_ratio": (
            strict_multi_pred_rows / n_rows
        ) if n_rows > 0 else 0.0,
        "strict_branch_rows": strict_branch_rows,
        "strict_branch_row_ratio": (
            strict_branch_rows / n_rows
        ) if n_rows > 0 else 0.0,
        "strict_chain_count": strict_chain_count,
        "strict_chain_row_coverage": strict_chain_row_coverage,
        "strict_chain_row_coverage_ratio": (
            strict_chain_row_coverage / n_rows
        ) if n_rows > 0 else 0.0,
        "max_strict_chain_length": int(max_strict_chain_length),
        "avg_strict_chain_length": float(avg_strict_chain_length),
        "strict_chain_len_ge_2": strict_chain_len_ge_2,
        "strict_chain_len_ge_4": strict_chain_len_ge_4,
        "strict_chain_len_ge_8": strict_chain_len_ge_8,
        "strict_chain_len_ge_16": strict_chain_len_ge_16,
        "strict_chain_len_ge_32": strict_chain_len_ge_32,
        "strict_chain_len_ge_64": strict_chain_len_ge_64,
    }

    return stats, dumped


def build_frontier_safe_parent(graph: StrictLowerDependencyGraph):
    """
    Pick the unique deepest predecessor as the candidate scan parent.

    Candidate edges are later split by the chain-head boundary-known rule.
    Rows with tied deepest predecessors are conservatively rejected.
    """
    n_rows = graph.n_rows
    index_dtype = _index_dtype(n_rows)
    level = np.ones(n_rows, dtype=index_dtype)
    frontier_parent = np.full(n_rows, -1, dtype=graph.indices.dtype)
    invalid_multi_deepest_rows = 0

    for row in range(n_rows):
        start = int(graph.indptr[row])
        end = int(graph.indptr[row + 1])
        if start == end:
            continue

        max_pred_level = -1
        max_pred = -1
        max_pred_count = 0

        for entry in range(start, end):
            pred = int(graph.indices[entry])
            pred_level = int(level[pred])
            if pred_level > max_pred_level:
                max_pred_level = pred_level
                max_pred = pred
                max_pred_count = 1
            elif pred_level == max_pred_level:
                max_pred_count += 1

        level[row] = max_pred_level + 1
        if max_pred_count == 1:
            frontier_parent[row] = max_pred
        else:
            invalid_multi_deepest_rows += 1

    return level, frontier_parent, invalid_multi_deepest_rows


def _is_frontier_safe_boundary_head(row, frontier_parent, frontier_child_count):
    if frontier_child_count[row] == 0:
        return False

    parent = int(frontier_parent[row])
    if parent == -1:
        return True

    return frontier_child_count[parent] != 1


def _can_extend_frontier_safe_chain(
    graph: StrictLowerDependencyGraph,
    row: int,
    main_parent: int,
    chain_head: int,
):
    """
    A candidate child can stay in the current chain only if every non-main
    predecessor is already outside the current head boundary.
    """
    start = int(graph.indptr[row])
    end = int(graph.indptr[row + 1])
    for entry in range(start, end):
        pred = int(graph.indices[entry])
        if pred == main_parent:
            continue
        if pred >= chain_head:
            return False
    return True


def analyze_frontier_safe_chains_from_graph(
    graph: StrictLowerDependencyGraph,
    dump_chains=False,
):
    """Compute frontier-safe chains from candidate parent edges plus boundary cuts."""
    n_rows = graph.n_rows
    level, frontier_parent, invalid_multi_deepest_rows = build_frontier_safe_parent(
        graph
    )
    frontier_children = build_strict_child_arrays(frontier_parent)

    frontier_safe_eligible_rows = int(np.count_nonzero(frontier_parent != -1))
    frontier_safe_branch_rows = int(
        np.count_nonzero(frontier_children.strict_child_count > 1)
    )
    frontier_safe_max_level = int(np.max(level)) if n_rows > 0 else 0

    frontier_safe_boundary_cut_rows = 0
    frontier_safe_chain_count = 0
    frontier_safe_chain_row_coverage = 0
    max_frontier_safe_chain_length = 0
    frontier_safe_chain_len_ge_2 = 0
    frontier_safe_chain_len_ge_4 = 0
    frontier_safe_chain_len_ge_8 = 0
    frontier_safe_chain_len_ge_16 = 0
    frontier_safe_chain_len_ge_32 = 0
    frontier_safe_chain_len_ge_64 = 0
    dumped = [] if dump_chains else None

    def record_chain(head, head_type, length, nodes):
        nonlocal frontier_safe_chain_count
        nonlocal frontier_safe_chain_row_coverage
        nonlocal max_frontier_safe_chain_length
        nonlocal frontier_safe_chain_len_ge_2
        nonlocal frontier_safe_chain_len_ge_4
        nonlocal frontier_safe_chain_len_ge_8
        nonlocal frontier_safe_chain_len_ge_16
        nonlocal frontier_safe_chain_len_ge_32
        nonlocal frontier_safe_chain_len_ge_64

        if length < 2:
            return

        frontier_safe_chain_count += 1
        frontier_safe_chain_row_coverage += length
        max_frontier_safe_chain_length = max(max_frontier_safe_chain_length, length)

        if length >= 2:
            frontier_safe_chain_len_ge_2 += 1
        if length >= 4:
            frontier_safe_chain_len_ge_4 += 1
        if length >= 8:
            frontier_safe_chain_len_ge_8 += 1
        if length >= 16:
            frontier_safe_chain_len_ge_16 += 1
        if length >= 32:
            frontier_safe_chain_len_ge_32 += 1
        if length >= 64:
            frontier_safe_chain_len_ge_64 += 1

        if dump_chains:
            dumped.append(
                {
                    "head": head,
                    "head_type": head_type,
                    "head_pred_count": int(graph.pred_count[head]),
                    "length": length,
                    "nodes": nodes,
                }
            )

    for head in range(n_rows):
        if not _is_frontier_safe_boundary_head(
            head, frontier_parent, frontier_children.strict_child_count
        ):
            continue

        chain_head = head
        chain_head_type = "source" if frontier_parent[head] == -1 else "boundary_head"
        length = 1
        current = head
        nodes = [head] if dump_chains else None

        while frontier_children.strict_child_count[current] == 1:
            child = int(frontier_children.unique_strict_child[current])
            if child == -1:
                raise RuntimeError(
                    f"Missing unique frontier-safe child for row {current}"
                )

            if not _can_extend_frontier_safe_chain(
                graph, child, current, chain_head
            ):
                frontier_safe_boundary_cut_rows += 1
                record_chain(chain_head, chain_head_type, length, nodes)
                chain_head = child
                chain_head_type = "boundary_cut_head"
                current = child
                length = 1
                nodes = [child] if dump_chains else None
                continue

            current = child
            length += 1
            if dump_chains:
                nodes.append(current)

        record_chain(chain_head, chain_head_type, length, nodes)

    avg_frontier_safe_chain_length = (
        frontier_safe_chain_row_coverage / frontier_safe_chain_count
        if frontier_safe_chain_count > 0
        else 0.0
    )

    stats = {
        "frontier_safe_max_level": frontier_safe_max_level,
        "frontier_safe_eligible_rows": frontier_safe_eligible_rows,
        "frontier_safe_eligible_row_ratio": (
            frontier_safe_eligible_rows / n_rows
        ) if n_rows > 0 else 0.0,
        "frontier_safe_invalid_multi_deepest_rows": int(
            invalid_multi_deepest_rows
        ),
        "frontier_safe_invalid_multi_deepest_row_ratio": (
            invalid_multi_deepest_rows / n_rows
        ) if n_rows > 0 else 0.0,
        "frontier_safe_branch_rows": frontier_safe_branch_rows,
        "frontier_safe_branch_row_ratio": (
            frontier_safe_branch_rows / n_rows
        ) if n_rows > 0 else 0.0,
        "frontier_safe_boundary_cut_rows": frontier_safe_boundary_cut_rows,
        "frontier_safe_chain_count": frontier_safe_chain_count,
        "frontier_safe_chain_row_coverage": frontier_safe_chain_row_coverage,
        "frontier_safe_chain_row_coverage_ratio": (
            frontier_safe_chain_row_coverage / n_rows
        ) if n_rows > 0 else 0.0,
        "max_frontier_safe_chain_length": int(max_frontier_safe_chain_length),
        "avg_frontier_safe_chain_length": float(avg_frontier_safe_chain_length),
        "frontier_safe_chain_len_ge_2": frontier_safe_chain_len_ge_2,
        "frontier_safe_chain_len_ge_4": frontier_safe_chain_len_ge_4,
        "frontier_safe_chain_len_ge_8": frontier_safe_chain_len_ge_8,
        "frontier_safe_chain_len_ge_16": frontier_safe_chain_len_ge_16,
        "frontier_safe_chain_len_ge_32": frontier_safe_chain_len_ge_32,
        "frontier_safe_chain_len_ge_64": frontier_safe_chain_len_ge_64,
    }

    return stats, dumped


def matrix_name_for_path(path: Path):
    return path.stem


def spcg_factor_matrix_name(
    source_path: Path,
    method: str,
    fill_factor: float,
    drop_tol: float | None,
    sparsify_percentage: float,
):
    drop_tag = "default" if drop_tol is None else f"{drop_tol:g}"
    return (
        f"spcg_{method}_l_{source_path.stem}"
        f"_fill{fill_factor:g}_drop{drop_tag}_sp{sparsify_percentage:g}"
    )


def analyze_dependencies(
    deps: StrictLowerDependencies,
    matrix_name: str,
    dump_chains=False,
    max_dump_rows=DEFAULT_MAX_DUMP_ROWS,
    include_frontier_safe=False,
    dependency_graph: StrictLowerDependencyGraph | None = None,
):
    n_rows = int(deps.n_rows)
    strict_lower_nnz = int(deps.strict_lower_nnz)
    nnz = strict_lower_nnz + n_rows

    if dump_chains and n_rows > max_dump_rows:
        raise ValueError(
            f"--dump-chains is intended for small matrices; {matrix_name} has "
            f"{n_rows} rows, limit is {max_dump_rows}"
        )

    strict_parent = build_strict_parent(deps.pred_count, deps.first_pred)
    strict_children = build_strict_child_arrays(strict_parent)
    strict_stats, dumped_chains = analyze_strict_chains_from_arrays(
        deps.pred_count,
        strict_parent,
        strict_children.strict_child_count,
        strict_children.unique_strict_child,
        dump_chains=dump_chains,
    )

    result = {
        "matrix_name": matrix_name,
        "n_rows": n_rows,
        "strict_lower_nnz": strict_lower_nnz,
        "nnz": nnz,
        **strict_stats,
    }

    if dump_chains:
        result["strict_chains"] = dumped_chains

    if include_frontier_safe:
        if dependency_graph is None:
            raise ValueError("Frontier-safe analysis requires a dependency graph")
        frontier_stats, dumped_frontier_chains = analyze_frontier_safe_chains_from_graph(
            dependency_graph,
            dump_chains=dump_chains,
        )
        result.update(frontier_stats)
        if dump_chains:
            result["frontier_safe_chains"] = dumped_frontier_chains

    return result


def analyze_matrix(
    path: Path,
    triangle="lower",
    dump_chains=False,
    max_dump_rows=DEFAULT_MAX_DUMP_ROWS,
    include_frontier_safe=False,
):
    if include_frontier_safe:
        deps = read_strict_lower_dependency_graph_mtx(path, triangle=triangle)
    else:
        deps = read_strict_lower_dependencies_mtx(path, triangle=triangle)
    return analyze_dependencies(
        deps,
        matrix_name_for_path(path),
        dump_chains=dump_chains,
        max_dump_rows=max_dump_rows,
        include_frontier_safe=include_frontier_safe,
        dependency_graph=deps if include_frontier_safe else None,
    )


def _require_scipy_for_spcg_factors():
    try:
        import scipy.io as scipy_io
        import scipy.sparse as scipy_sparse
        import scipy.sparse.linalg as scipy_linalg
    except ImportError as exc:
        raise RuntimeError(
            "SPCG factor generation requires scipy. Install scipy or use "
            "--source factor-l with an existing L factor .mtx file."
        ) from exc

    return scipy_io, scipy_sparse, scipy_linalg


def _spcg_sparsify_matrix(matrix, percentage: float):
    if percentage <= 0:
        return matrix

    matrix = matrix.tocoo(copy=True)
    if matrix.nnz == 0:
        return matrix.tocsr()

    threshold_index = int(len(matrix.data) * percentage)
    threshold_index = min(max(threshold_index, 0), len(matrix.data) - 1)
    threshold = np.sort(np.abs(matrix.data))[threshold_index]

    keep = (matrix.row == matrix.col) | (np.abs(matrix.data) > threshold)
    return type(matrix)((matrix.data[keep], (matrix.row[keep], matrix.col[keep])), shape=matrix.shape).tocsr()


def generate_spcg_lu_factors(
    matrix_path: Path,
    factor_dir: Path,
    method: str,
    fill_factor: float,
    drop_tol: float | None,
    sparsify_percentage: float,
):
    """Generate SPCG-style SuperLU L/U factors and export them as Matrix Market."""
    scipy_io, scipy_sparse, scipy_linalg = _require_scipy_for_spcg_factors()

    matrix = scipy_io.mmread(str(matrix_path))
    if not hasattr(matrix, "tocsc"):
        matrix = scipy_sparse.coo_matrix(matrix)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix is not square: {matrix_path.name}, shape={matrix.shape}")

    matrix = _spcg_sparsify_matrix(matrix, sparsify_percentage)
    matrix_csc = matrix.tocsc()

    if method == "splu":
        factors = scipy_linalg.splu(
            matrix_csc,
            permc_spec="NATURAL",
            diag_pivot_thresh=1e-15,
        )
    elif method == "spilu":
        kwargs = {
            "fill_factor": fill_factor,
            "permc_spec": "NATURAL",
            "diag_pivot_thresh": 1e-15,
        }
        if drop_tol is not None:
            kwargs["drop_tol"] = drop_tol
        factors = scipy_linalg.spilu(matrix_csc, **kwargs)
    else:
        raise ValueError(f"Unsupported SPCG factorization method: {method}")

    matrix_name = spcg_factor_matrix_name(
        matrix_path,
        method=method,
        fill_factor=fill_factor,
        drop_tol=drop_tol,
        sparsify_percentage=sparsify_percentage,
    )

    matrix_factor_dir = factor_dir / matrix_path.stem
    matrix_factor_dir.mkdir(parents=True, exist_ok=True)
    l_path = matrix_factor_dir / f"{matrix_name}.mtx"
    u_path = matrix_factor_dir / f"{matrix_name.replace('_l_', '_u_', 1)}.mtx"
    scipy_io.mmwrite(str(l_path), factors.L)
    scipy_io.mmwrite(str(u_path), factors.U)

    return l_path, u_path, matrix_name


def analyze_spcg_factor_l(
    matrix_path: Path,
    factor_dir: Path,
    method: str,
    fill_factor: float,
    drop_tol: float | None,
    sparsify_percentage: float,
    triangle="lower",
    dump_chains=False,
    max_dump_rows=DEFAULT_MAX_DUMP_ROWS,
    include_frontier_safe=False,
):
    l_path, _, matrix_name = generate_spcg_lu_factors(
        matrix_path=matrix_path,
        factor_dir=factor_dir,
        method=method,
        fill_factor=fill_factor,
        drop_tol=drop_tol,
        sparsify_percentage=sparsify_percentage,
    )
    if include_frontier_safe:
        deps = read_strict_lower_dependency_graph_mtx(l_path, triangle=triangle)
    else:
        deps = read_strict_lower_dependencies_mtx(l_path, triangle=triangle)
    return analyze_dependencies(
        deps,
        matrix_name,
        dump_chains=dump_chains,
        max_dump_rows=max_dump_rows,
        include_frontier_safe=include_frontier_safe,
        dependency_graph=deps if include_frontier_safe else None,
    )


def collect_mtx_files(input_path: Path, source="matrix-lower"):
    if input_path.is_file():
        return [input_path]

    files = sorted(input_path.rglob("*.mtx"))
    if source == "factor-l":
        return [
            path
            for path in files
            if "_l_" in path.stem.lower()
            or path.stem.lower().endswith("_l")
            or "l_factor" in path.stem.lower()
        ]
    return files


def resolve_input_path(input_arg: str):
    input_path = Path(input_arg).expanduser()
    if input_path.exists():
        return input_path

    repo_relative = REPO_ROOT / input_path
    if repo_relative.exists():
        return repo_relative

    matrix_relative = MATRIX_DIR / input_path
    if matrix_relative.exists():
        return matrix_relative

    if input_path.suffix != ".mtx":
        name = input_path.name
        candidates = [
            REAL_MATRIX_DIR / name / f"{name}.mtx",
            REAL_MATRIX_DIR / f"{name}.mtx",
            REGRESSION_MATRIX_DIR / f"{name}.mtx",
            MATRIX_DIR / name / f"{name}.mtx",
            MATRIX_DIR / f"{name}.mtx",
            SPCG_FACTOR_DIR / name / f"{name}.mtx",
            SPCG_FACTOR_DIR / f"{name}.mtx",
            REPO_ROOT / name / f"{name}.mtx",
            REPO_ROOT / f"{name}.mtx",
            Path.cwd() / name / f"{name}.mtx",
            Path.cwd() / f"{name}.mtx",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    return input_path


def resolve_repo_relative_path(path_arg: str | None):
    if path_arg is None:
        return None

    path = Path(path_arg).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def csv_fields_for(include_frontier_safe: bool):
    if include_frontier_safe:
        return CSV_FIELDS + FRONTIER_SAFE_CSV_FIELDS
    return CSV_FIELDS


def default_output_path(source: str, include_frontier_safe: bool):
    if include_frontier_safe:
        if source == "matrix-lower":
            return DEFAULT_FRONTIER_SAFE_MATRIX_LOWER_OUT
        return DEFAULT_FRONTIER_SAFE_SPCG_FACTOR_OUT

    if source == "matrix-lower":
        return DEFAULT_MATRIX_LOWER_OUT
    return DEFAULT_SPCG_FACTOR_OUT


def is_label_row(row, field_labels):
    return row.get("matrix_name") == field_labels["matrix_name"]


def read_existing_result_rows(output_path: Path, csv_fields, field_labels):
    if not output_path.exists() or output_path.stat().st_size == 0:
        return []

    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != csv_fields:
            raise ValueError(
                f"Existing CSV schema does not match current fields: {output_path}"
            )
        return [row for row in reader if not is_label_row(row, field_labels)]


def write_upsert_results_csv(output_path: Path, new_rows, csv_fields):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    field_labels = {field: CSV_FIELD_LABELS[field] for field in csv_fields}
    rows_by_name = {
        row["matrix_name"]: row
        for row in read_existing_result_rows(output_path, csv_fields, field_labels)
    }
    row_order = list(rows_by_name)

    for row in new_rows:
        matrix_name = row["matrix_name"]
        if matrix_name not in rows_by_name:
            row_order.append(matrix_name)
        rows_by_name[matrix_name] = row

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for matrix_name in row_order:
            row = rows_by_name[matrix_name]
            writer.writerow({field: row[field] for field in csv_fields})
        writer.writerow(field_labels)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze strict single-predecessor chains for SpTRSV matrices."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="mawi_201512020030",
        help=(
            "Input .mtx file, directory, or matrix name. A bare name like "
            "'mawi_201512020030' resolves to "
            "data/matrices/real/mawi_201512020030/mawi_201512020030.mtx"
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output CSV path. If omitted, the default is chosen from "
            "the data source and analysis method."
        ),
    )
    parser.add_argument(
        "--small-test",
        action="store_true",
        help=(
            f"Use {DEFAULT_SMALL_TEST_INPUT} and write to "
            f"{DEFAULT_SMALL_TEST_OUT} or {DEFAULT_SMALL_TEST_FRONTIER_OUT}"
        ),
    )
    parser.add_argument(
        "--dump-chains",
        action="store_true",
        help="Print each extracted chain as JSON lines",
    )
    parser.add_argument(
        "--dump-chains-out",
        default=None,
        help="Optional JSONL output path for dumped strict chains",
    )
    parser.add_argument(
        "--dump-chains-max-rows",
        type=int,
        default=DEFAULT_MAX_DUMP_ROWS,
        help=(
            "Maximum matrix rows allowed with --dump-chains "
            f"(default: {DEFAULT_MAX_DUMP_ROWS})"
        ),
    )
    parser.add_argument(
        "--triangle",
        default="lower",
        choices=["lower"],
        help="Triangle type after preprocessing (currently only lower is supported)",
    )
    parser.add_argument(
        "--frontier-safe",
        action="store_true",
        help=(
            "Also compute frontier-safe chains: unique deepest predecessors "
            "first build candidate parent edges, then chain-head boundary "
            "checks split unsafe candidate chains."
        ),
    )
    parser.add_argument(
        "--source",
        default="matrix-lower",
        choices=["matrix-lower", "factor-l", "spcg-ilu-l"],
        help=(
            "Data source. matrix-lower streams an original matrix and extracts "
            "strict lower dependencies; factor-l analyzes an existing L factor; "
            "spcg-ilu-l generates SPCG-style SuperLU L/U factors and analyzes L."
        ),
    )
    parser.add_argument(
        "--factor-dir",
        default=str(DEFAULT_FACTOR_DIR),
        help=f"Directory for generated SPCG-style L/U factors (default: {DEFAULT_FACTOR_DIR})",
    )
    parser.add_argument(
        "--spcg-method",
        default="spilu",
        choices=["spilu", "splu"],
        help="SPCG-style factorization method used by --source spcg-ilu-l",
    )
    parser.add_argument(
        "--fill-factor",
        type=float,
        default=10.0,
        help="SuperLU fill_factor for --spcg-method spilu (default: 10.0)",
    )
    parser.add_argument(
        "--drop-tol",
        type=float,
        default=None,
        help="Optional SuperLU drop_tol for --spcg-method spilu",
    )
    parser.add_argument(
        "--spcg-sparsify-percentage",
        type=float,
        default=0.0,
        help=(
            "Optional SPCG-style value-threshold sparsification percentage before "
            "factorization, for --source spcg-ilu-l (default: 0.0)"
        ),
    )
    args = parser.parse_args()

    if args.spcg_sparsify_percentage < 0 or args.spcg_sparsify_percentage >= 1:
        raise ValueError("--spcg-sparsify-percentage must be in [0, 1)")

    if args.small_test:
        input_path = DEFAULT_SMALL_TEST_INPUT
        output_path = (
            resolve_repo_relative_path(args.out)
            if args.out is not None
            else (
                DEFAULT_SMALL_TEST_FRONTIER_OUT
                if args.frontier_safe
                else DEFAULT_SMALL_TEST_OUT
            )
        )
    else:
        input_path = resolve_input_path(args.input)
        if args.out is not None:
            output_path = resolve_repo_relative_path(args.out)
        else:
            output_path = default_output_path(args.source, args.frontier_safe)
    dump_chains_out_path = resolve_repo_relative_path(args.dump_chains_out)
    factor_dir = resolve_repo_relative_path(args.factor_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    mtx_files = collect_mtx_files(input_path, source=args.source)
    if not mtx_files:
        print(f"No .mtx files found in {input_path}")

    results = []
    dump_lines = []
    for mtx in mtx_files:
        try:
            if args.source in {"matrix-lower", "factor-l"}:
                res = analyze_matrix(
                    mtx,
                    triangle=args.triangle,
                    dump_chains=args.dump_chains,
                    max_dump_rows=args.dump_chains_max_rows,
                    include_frontier_safe=args.frontier_safe,
                )
            else:
                res = analyze_spcg_factor_l(
                    mtx,
                    factor_dir=factor_dir,
                    method=args.spcg_method,
                    fill_factor=args.fill_factor,
                    drop_tol=args.drop_tol,
                    sparsify_percentage=args.spcg_sparsify_percentage,
                    triangle=args.triangle,
                    dump_chains=args.dump_chains,
                    max_dump_rows=args.dump_chains_max_rows,
                    include_frontier_safe=args.frontier_safe,
                )
            results.append(res)
            if args.dump_chains:
                chain_dump = {
                    "matrix_name": res["matrix_name"],
                    "strict_chains": res.get("strict_chains", []),
                }
                if args.frontier_safe:
                    chain_dump["frontier_safe_chains"] = res.get(
                        "frontier_safe_chains", []
                    )
                chain_dump_line = json.dumps(chain_dump, ensure_ascii=False)
                print(chain_dump_line)
                dump_lines.append(chain_dump_line)
        except Exception as exc:
            print(f"[WARN] Skip {mtx.name}: {exc}")

    write_upsert_results_csv(output_path, results, csv_fields_for(args.frontier_safe))

    if dump_chains_out_path is not None:
        dump_chains_out_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_chains_out_path.open("w", encoding="utf-8") as f:
            for line in dump_lines:
                f.write(line + "\n")

    print(f"Done. Processed {len(results)} matrices. Output: {output_path}")


if __name__ == "__main__":
    main()

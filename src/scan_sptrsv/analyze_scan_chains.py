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

FRONTIER_SAFE_CHAIN_ENABLED = False
DEFAULT_MAX_DUMP_ROWS = 1_000_000
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
RESULTS_DIR = REPO_ROOT / "results"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DEFAULT_BIGTEST_OUT = RESULTS_DIR / "bigtest_end.csv"
DEFAULT_SMALL_TEST_INPUT = DATASETS_DIR / "datasets1"
DEFAULT_SMALL_TEST_OUT = EXPERIMENTS_DIR / "scan_chain_stats_t01_t05_test.csv"

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


def analyze_frontier_safe_chains():
    """
    Reserved parallel analysis path for future frontier-safe chains.

    This is intentionally disabled for now so strict-chain verification can
    remain stable. When enabled later, this function should derive a second
    chain family from the same predecessor graph without changing strict stats.
    """
    return {
        "enabled": False,
        "status": "reserved",
        "reason": "frontier-safe chain analysis is scaffolded but not enabled yet",
    }


def matrix_name_for_path(path: Path):
    return path.stem


def analyze_matrix(
    path: Path,
    triangle="lower",
    dump_chains=False,
    max_dump_rows=DEFAULT_MAX_DUMP_ROWS,
):
    deps = read_strict_lower_dependencies_mtx(path, triangle=triangle)

    n_rows = int(deps.n_rows)
    strict_lower_nnz = int(deps.strict_lower_nnz)
    nnz = strict_lower_nnz + n_rows

    if dump_chains and n_rows > max_dump_rows:
        raise ValueError(
            f"--dump-chains is intended for small matrices; {path.name} has "
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
        "matrix_name": matrix_name_for_path(path),
        "n_rows": n_rows,
        "strict_lower_nnz": strict_lower_nnz,
        "nnz": nnz,
        **strict_stats,
    }

    if dump_chains:
        result["strict_chains"] = dumped_chains

    if FRONTIER_SAFE_CHAIN_ENABLED:
        result["frontier_safe_chain_analysis"] = analyze_frontier_safe_chains()

    return result


def collect_mtx_files(input_path: Path):
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.mtx"))


def resolve_input_path(input_arg: str):
    input_path = Path(input_arg).expanduser()
    if input_path.exists():
        return input_path

    repo_relative = REPO_ROOT / input_path
    if repo_relative.exists():
        return repo_relative

    datasets_relative = DATASETS_DIR / input_path
    if datasets_relative.exists():
        return datasets_relative

    if input_path.suffix != ".mtx":
        name = input_path.name
        candidates = [
            DATASETS_DIR / name / f"{name}.mtx",
            DATASETS_DIR / f"{name}.mtx",
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


def is_label_row(row):
    return row.get("matrix_name") == CSV_FIELD_LABELS["matrix_name"]


def read_existing_result_rows(output_path: Path):
    if not output_path.exists() or output_path.stat().st_size == 0:
        return []

    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(
                f"Existing CSV schema does not match current fields: {output_path}"
            )
        return [row for row in reader if not is_label_row(row)]


def write_upsert_results_csv(output_path: Path, new_rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_name = {
        row["matrix_name"]: row for row in read_existing_result_rows(output_path)
    }
    row_order = list(rows_by_name)

    for row in new_rows:
        matrix_name = row["matrix_name"]
        if matrix_name not in rows_by_name:
            row_order.append(matrix_name)
        rows_by_name[matrix_name] = row

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for matrix_name in row_order:
            row = rows_by_name[matrix_name]
            writer.writerow({field: row[field] for field in CSV_FIELDS})
        writer.writerow({field: CSV_FIELD_LABELS[field] for field in CSV_FIELDS})


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
            "datasets/mawi_201512020030/mawi_201512020030.mtx"
        ),
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_BIGTEST_OUT),
        help=f"Output CSV path (default: {DEFAULT_BIGTEST_OUT})",
    )
    parser.add_argument(
        "--small-test",
        action="store_true",
        help=(
            "Use datasets/datasets1 and write to "
            "experiments/scan_chain_stats_t01_t05_test.csv"
        ),
    )
    parser.add_argument(
        "--dump-chains",
        action="store_true",
        help="Print each extracted strict chain as JSON lines",
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
    args = parser.parse_args()

    if args.small_test:
        input_path = DEFAULT_SMALL_TEST_INPUT
        output_path = DEFAULT_SMALL_TEST_OUT
    else:
        input_path = resolve_input_path(args.input)
        output_path = resolve_repo_relative_path(args.out)
    dump_chains_out_path = resolve_repo_relative_path(args.dump_chains_out)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    mtx_files = collect_mtx_files(input_path)
    if not mtx_files:
        print(f"No .mtx files found in {input_path}")

    results = []
    dump_lines = []
    for mtx in mtx_files:
        try:
            res = analyze_matrix(
                mtx,
                triangle=args.triangle,
                dump_chains=args.dump_chains,
                max_dump_rows=args.dump_chains_max_rows,
            )
            results.append(res)
            if args.dump_chains:
                chain_dump = {
                    "matrix_name": res["matrix_name"],
                    "strict_chains": res.get("strict_chains", []),
                }
                chain_dump_line = json.dumps(chain_dump, ensure_ascii=False)
                print(chain_dump_line)
                dump_lines.append(chain_dump_line)
        except Exception as exc:
            print(f"[WARN] Skip {mtx.name}: {exc}")

    write_upsert_results_csv(output_path, results)

    if dump_chains_out_path is not None:
        dump_chains_out_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_chains_out_path.open("w", encoding="utf-8") as f:
            for line in dump_lines:
                f.write(line + "\n")

    print(f"Done. Processed {len(results)} matrices. Output: {output_path}")


if __name__ == "__main__":
    main()

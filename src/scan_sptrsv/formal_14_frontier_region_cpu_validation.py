#!/usr/bin/env python3
"""CPU numerical validation for formal frontier-safe bounded-width regions.

This is a validation-only experiment. It does not drop entries, regenerate
factors, download matrices, optimize performance, or implement a GPU executor.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scan_sptrsv import analyze_scan_chains as base
from scan_sptrsv.formal_14_experiment import (
    DEFAULT_CFD_MANIFEST,
    DEFAULT_SPD_MANIFEST,
    load_formal_factors,
)
from scan_sptrsv.formal_14_frontier_relaxed_region import (
    Region,
    greedy_bounded_width_regions,
    summarize_regions,
)


WIDTHS = (2, 4)
ERROR_TOL = 1e-9
DEFAULT_OUT = (
    base.RESULTS_DIR
    / "drop_experiments"
    / "formal_14_frontier_region_cpu_validation.csv"
)
DEFAULT_SUMMARY = (
    base.RESULTS_DIR
    / "drop_experiments"
    / "formal_14_frontier_region_cpu_validation_summary.txt"
)


CSV_FIELDS = [
    "status",
    "error",
    "matrix",
    "group",
    "w",
    "n",
    "nnz_L",
    "region_count",
    "frontier_safe_region_count",
    "validated_region_count",
    "failed_region_count",
    "covered_rows",
    "covered_row_ratio",
    "structural_ge8_row_coverage",
    "coverage_retention",
    "baseline_max_abs_error",
    "baseline_rel_residual",
    "max_abs_error_seq",
    "max_abs_error_affine",
    "max_abs_error_scan_sim",
    "max_rel_error_scan_sim",
    "residual_error",
    "external_unresolved_count",
    "avg_external_nnz_per_row",
    "avg_internal_nnz_per_row",
    "verdict",
]


@dataclass(frozen=True)
class NumericLowerFactor:
    csr: object
    diag: np.ndarray
    n: int
    nnz: int


@dataclass
class RegionValidationResult:
    frontier_safe: bool
    external_unresolved_count: int
    rows: int
    external_nnz: int
    internal_nnz: int
    max_abs_error_seq: float
    max_abs_error_affine: float
    max_abs_error_scan_sim: float
    max_rel_error_scan_sim: float
    failed: bool
    error: str = ""


def _float(row: dict, field: str) -> float:
    value = row.get(field, "")
    if value == "":
        return 0.0
    return float(value)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def require_scipy():
    try:
        import scipy.io as scipy_io
        import scipy.sparse as scipy_sparse
    except ImportError as exc:
        raise RuntimeError("SciPy is required for CPU numerical validation") from exc
    return scipy_io, scipy_sparse


def read_numeric_lower_factor(path: Path) -> NumericLowerFactor:
    scipy_io, scipy_sparse = require_scipy()
    matrix = scipy_io.mmread(str(path))
    if not hasattr(matrix, "tocsr"):
        matrix = scipy_sparse.coo_matrix(matrix)
    matrix = matrix.tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Factor is not square: {path}, shape={matrix.shape}")
    diag = matrix.diagonal().astype(np.float64, copy=False)
    if np.any(np.abs(diag) == 0):
        zero_count = int(np.count_nonzero(np.abs(diag) == 0))
        raise ValueError(f"Zero diagonal entries in {path.name}: {zero_count}")
    return NumericLowerFactor(
        csr=matrix,
        diag=diag,
        n=matrix.shape[0],
        nnz=matrix.nnz,
    )


def solve_lower_sequential(factor: NumericLowerFactor, b: np.ndarray) -> np.ndarray:
    matrix = factor.csr
    x = np.zeros(factor.n, dtype=np.float64)
    for row in range(factor.n):
        start = int(matrix.indptr[row])
        end = int(matrix.indptr[row + 1])
        lower_sum = 0.0
        for entry in range(start, end):
            col = int(matrix.indices[entry])
            if col < row:
                lower_sum += float(matrix.data[entry]) * x[col]
        x[row] = (b[row] - lower_sum) / factor.diag[row]
    return x


def lower_row_entries(
    factor: NumericLowerFactor,
    row: int,
) -> tuple[list[tuple[int, float]], float]:
    matrix = factor.csr
    start = int(matrix.indptr[row])
    end = int(matrix.indptr[row + 1])
    lower_entries: list[tuple[int, float]] = []
    diag = factor.diag[row]
    for entry in range(start, end):
        col = int(matrix.indices[entry])
        value = float(matrix.data[entry])
        if col < row:
            lower_entries.append((col, value))
    return lower_entries, float(diag)


def matrix_seed(matrix_name: str) -> int:
    return zlib.crc32(matrix_name.encode("utf-8")) & 0xFFFFFFFF


def baseline_for_factor(factor, numeric: NumericLowerFactor):
    rng = np.random.default_rng(matrix_seed(factor.matrix_name))
    x_true = rng.standard_normal(numeric.n)
    b = numeric.csr @ x_true
    x_ref = solve_lower_sequential(numeric, b)
    baseline_abs = float(np.max(np.abs(x_ref - x_true))) if numeric.n else 0.0
    residual = numeric.csr @ x_ref - b
    residual_rel = float(
        np.linalg.norm(residual) / max(np.linalg.norm(b), 1e-300)
    )
    return x_true, b, x_ref, baseline_abs, residual_rel


def effective_regions_for_width(
    graph: base.StrictLowerDependencyGraph,
    levels: np.ndarray,
    width: int,
) -> tuple[tuple[Region, ...], float]:
    regions = greedy_bounded_width_regions(graph, levels, width)
    structural = summarize_regions(
        regions,
        levels,
        graph.n_rows,
        int(np.max(levels)) if graph.n_rows else 0,
    )
    return tuple(region for region in regions if region.length >= 8), structural.ge8_row_coverage


def build_region_successors(
    factor: NumericLowerFactor,
    region: Region,
) -> tuple[dict[int, int], int, int, int]:
    last_successor: dict[int, int] = {}
    external_unresolved = 0
    internal_nnz = 0
    external_nnz = 0
    for row in range(region.start, region.end + 1):
        lower_entries, _ = lower_row_entries(factor, row)
        for pred, _ in lower_entries:
            if pred < region.start:
                external_nnz += 1
            elif region.start <= pred < row:
                internal_nnz += 1
                last_successor[pred] = row
            else:
                external_unresolved += 1
    return last_successor, external_unresolved, external_nnz, internal_nnz


def validate_region(
    factor: NumericLowerFactor,
    region: Region,
    width: int,
    b: np.ndarray,
    x_ref: np.ndarray,
) -> RegionValidationResult:
    try:
        last_successor, unresolved, external_nnz, internal_nnz = build_region_successors(
            factor,
            region,
        )
        if unresolved:
            return RegionValidationResult(
                frontier_safe=False,
                external_unresolved_count=unresolved,
                rows=region.length,
                external_nnz=external_nnz,
                internal_nnz=internal_nnz,
                max_abs_error_seq=0.0,
                max_abs_error_affine=0.0,
                max_abs_error_scan_sim=0.0,
                max_rel_error_scan_sim=0.0,
                failed=True,
                error="external/internal split has unresolved dependencies",
            )

        state_rows: list[int] = []
        state = np.zeros(width, dtype=np.float64)
        x_seq: dict[int, float] = {}
        transitions = []
        max_seq = 0.0
        max_affine = 0.0

        for row in range(region.start, region.end + 1):
            lower_entries, diag = lower_row_entries(factor, row)
            state_pos = {state_row: pos for pos, state_row in enumerate(state_rows)}
            coeff = np.zeros(width, dtype=np.float64)
            rhs = float(b[row])
            seq_sum = 0.0

            for pred, value in lower_entries:
                if pred < region.start:
                    rhs -= value * x_ref[pred]
                elif region.start <= pred < row:
                    if pred not in state_pos or pred not in x_seq:
                        raise RuntimeError(
                            f"Internal predecessor {pred} unavailable before row {row}"
                        )
                    coeff[state_pos[pred]] -= value / diag
                    seq_sum += value * x_seq[pred]
                else:
                    raise RuntimeError(
                        f"Unresolved dependency {pred} for row {row}"
                    )

            q = rhs / diag
            x_row_seq = q - (seq_sum / diag)
            x_row_affine = float(coeff @ state + q)
            max_seq = max(max_seq, abs(x_row_seq - x_ref[row]))
            max_affine = max(max_affine, abs(x_row_affine - x_ref[row]))
            x_seq[row] = x_row_seq

            next_rows = [
                state_row for state_row in state_rows
                if last_successor.get(state_row, -1) > row
            ]
            if last_successor.get(row, -1) > row:
                next_rows.append(row)
            next_rows.sort()
            if len(next_rows) > width:
                raise RuntimeError(
                    f"Live state width exceeded: {len(next_rows)} > {width}"
                )

            transition_a = np.zeros((width, width), dtype=np.float64)
            transition_c = np.zeros(width, dtype=np.float64)
            for next_pos, state_row in enumerate(next_rows):
                if state_row == row:
                    transition_a[next_pos, :] = coeff
                    transition_c[next_pos] = q
                else:
                    transition_a[next_pos, state_pos[state_row]] = 1.0

            transitions.append((transition_a, transition_c, coeff.copy(), q, row))
            state = transition_a @ state + transition_c
            state_rows = next_rows

        affine_state = np.zeros(width, dtype=np.float64)
        max_affine_replay = 0.0
        for transition_a, transition_c, coeff, q, row in transitions:
            x_row = float(coeff @ affine_state + q)
            max_affine_replay = max(max_affine_replay, abs(x_row - x_ref[row]))
            affine_state = transition_a @ affine_state + transition_c
        max_affine = max(max_affine, max_affine_replay)

        prefix_a = np.eye(width, dtype=np.float64)
        prefix_c = np.zeros(width, dtype=np.float64)
        max_scan = 0.0
        max_rel_scan = 0.0
        for transition_a, transition_c, coeff, q, row in transitions:
            state_before = prefix_c
            x_row = float(coeff @ state_before + q)
            abs_error = abs(x_row - x_ref[row])
            rel_error = abs_error / max(abs(x_ref[row]), 1e-300)
            max_scan = max(max_scan, abs_error)
            max_rel_scan = max(max_rel_scan, rel_error)
            prefix_a, prefix_c = (
                transition_a @ prefix_a,
                transition_a @ prefix_c + transition_c,
            )

        return RegionValidationResult(
            frontier_safe=True,
            external_unresolved_count=0,
            rows=region.length,
            external_nnz=external_nnz,
            internal_nnz=internal_nnz,
            max_abs_error_seq=max_seq,
            max_abs_error_affine=max_affine,
            max_abs_error_scan_sim=max_scan,
            max_rel_error_scan_sim=max_rel_scan,
            failed=False,
        )
    except Exception as exc:
        return RegionValidationResult(
            frontier_safe=False,
            external_unresolved_count=1,
            rows=region.length,
            external_nnz=0,
            internal_nnz=0,
            max_abs_error_seq=0.0,
            max_abs_error_affine=0.0,
            max_abs_error_scan_sim=0.0,
            max_rel_error_scan_sim=0.0,
            failed=True,
            error=f"{type(exc).__name__}: {exc}",
        )


def validate_factor_width(
    formal_factor,
    numeric: NumericLowerFactor,
    graph: base.StrictLowerDependencyGraph,
    levels: np.ndarray,
    width: int,
    b: np.ndarray,
    x_ref: np.ndarray,
    baseline_abs_error: float,
    baseline_residual: float,
) -> dict:
    regions, structural_ge8_coverage = effective_regions_for_width(
        graph,
        levels,
        width,
    )
    results = [
        validate_region(numeric, region, width, b, x_ref)
        for region in regions
    ]

    frontier_safe = [result for result in results if result.frontier_safe]
    validated = [result for result in frontier_safe if not result.failed]
    failed = [result for result in results if result.failed]
    covered_rows = sum(result.rows for result in validated)
    external_nnz = sum(result.external_nnz for result in frontier_safe)
    internal_nnz = sum(result.internal_nnz for result in frontier_safe)
    frontier_rows = sum(result.rows for result in frontier_safe)
    max_abs_seq = max((result.max_abs_error_seq for result in validated), default=0.0)
    max_abs_affine = max(
        (result.max_abs_error_affine for result in validated),
        default=0.0,
    )
    max_abs_scan = max(
        (result.max_abs_error_scan_sim for result in validated),
        default=0.0,
    )
    max_rel_scan = max(
        (result.max_rel_error_scan_sim for result in validated),
        default=0.0,
    )
    unresolved = sum(result.external_unresolved_count for result in results)
    tolerance = max(ERROR_TOL, 100.0 * baseline_abs_error)
    coverage_ratio = covered_rows / numeric.n if numeric.n else 0.0
    coverage_retention = (
        coverage_ratio / structural_ge8_coverage
        if structural_ge8_coverage
        else 1.0
    )

    if failed:
        verdict = "fail-region-validation"
    elif max_abs_scan <= tolerance:
        verdict = "pass"
    else:
        verdict = "fail-scan-sim-error"

    return {
        "status": "ok",
        "error": "; ".join(result.error for result in failed if result.error),
        "matrix": formal_factor.matrix_name,
        "group": formal_factor.formal_group,
        "w": width,
        "n": numeric.n,
        "nnz_L": numeric.nnz,
        "region_count": len(regions),
        "frontier_safe_region_count": len(frontier_safe),
        "validated_region_count": len(validated),
        "failed_region_count": len(failed),
        "covered_rows": covered_rows,
        "covered_row_ratio": coverage_ratio,
        "structural_ge8_row_coverage": structural_ge8_coverage,
        "coverage_retention": coverage_retention,
        "baseline_max_abs_error": baseline_abs_error,
        "baseline_rel_residual": baseline_residual,
        "max_abs_error_seq": max_abs_seq,
        "max_abs_error_affine": max_abs_affine,
        "max_abs_error_scan_sim": max_abs_scan,
        "max_rel_error_scan_sim": max_rel_scan,
        "residual_error": baseline_residual,
        "external_unresolved_count": unresolved,
        "avg_external_nnz_per_row": (
            external_nnz / frontier_rows if frontier_rows else 0.0
        ),
        "avg_internal_nnz_per_row": (
            internal_nnz / frontier_rows if frontier_rows else 0.0
        ),
        "verdict": verdict,
    }


def failed_width_row(formal_factor, width: int, exc: Exception) -> dict:
    return {
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "matrix": formal_factor.matrix_name,
        "group": formal_factor.formal_group,
        "w": width,
        "verdict": "failed",
    }


def validate_factor(formal_factor) -> list[dict]:
    numeric = read_numeric_lower_factor(formal_factor.factor_l_path)
    graph = base.read_strict_lower_dependency_graph_mtx(formal_factor.factor_l_path)
    levels, _, _ = base.build_frontier_safe_parent(graph)
    _, b, x_ref, baseline_abs_error, baseline_residual = baseline_for_factor(
        formal_factor,
        numeric,
    )
    rows = []
    for width in WIDTHS:
        try:
            rows.append(
                validate_factor_width(
                    formal_factor=formal_factor,
                    numeric=numeric,
                    graph=graph,
                    levels=levels,
                    width=width,
                    b=b,
                    x_ref=x_ref,
                    baseline_abs_error=baseline_abs_error,
                    baseline_residual=baseline_residual,
                )
            )
        except Exception as exc:
            rows.append(failed_width_row(formal_factor, width, exc))
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def ok_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("status") == "ok"]


def pass_rows(rows: list[dict]) -> list[dict]:
    return [
        row for row in ok_rows(rows)
        if row.get("verdict") == "pass"
        and _float(row, "external_unresolved_count") == 0
        and _float(row, "coverage_retention") >= 0.90
        and _float(row, "max_abs_error_scan_sim") <= max(
            ERROR_TOL,
            100.0 * _float(row, "baseline_max_abs_error"),
        )
    ]


def support_rows(rows: list[dict], width: int) -> list[dict]:
    return [
        row for row in ok_rows(rows)
        if int(row["w"]) == width
        and _float(row, "structural_ge8_row_coverage") >= 0.10
    ]


def final_decision(rows: list[dict]) -> str:
    w4_support = support_rows(rows, 4)
    w4_pass = [row for row in w4_support if row in pass_rows(rows)]
    cfd_support = [row for row in w4_support if row.get("group") == "CFD"]
    cfd_pass = [row for row in cfd_support if row in pass_rows(rows)]

    if len(w4_support) >= 9 and len(w4_pass) >= 8 and len(cfd_pass) >= 4:
        return (
            "Final decision: B. CPU validation supports bounded-width scan-region "
            "equivalence; next step should be CPU executor/numerical convergence "
            "study, not GPU kernel."
        )
    return "Final decision: C. Not support scan-chain / scan-region direction."


def write_summary(summary_path: Path, rows: list[dict]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    ok = ok_rows(rows)
    passed = pass_rows(rows)
    w4_support = support_rows(rows, 4)
    w4_pass = [row for row in w4_support if row in passed]
    cfd_support = [row for row in w4_support if row.get("group") == "CFD"]
    cfd_pass = [row for row in cfd_support if row in passed]
    baseline_residual_warnings = [
        row for row in ok if _float(row, "baseline_rel_residual") > 1e-8
    ]
    baseline_inverse_warnings = [
        row
        for row in ok
        if _float(row, "baseline_rel_residual") <= 1e-8
        and _float(row, "baseline_max_abs_error") > 1e-6
    ]

    lines = [
        "Formal 14 frontier-region CPU validation summary",
        "",
        "Scope:",
        "- CPU numerical validation only.",
        "- No GPU kernel, no performance optimization, no dropping, no factor regeneration, no downloads.",
        "- Validates whether bounded-width regions can be represented as small-state affine scan transitions.",
        "",
        "Decision conditions:",
        "- w=4 support matrices need at least 8 / 9 pass.",
        "- CFD w=4 support matrices need at least 4 / 5 pass.",
        "- scan_sim max abs error must be <= 1e-9 or within 100x CPU baseline max abs error.",
        "- frontier-safe coverage retention must be >= 0.90.",
        "",
        f"rows: {len(rows)}",
        f"ok_rows: {len(ok)}",
        f"passed_rows: {len(passed)}",
        f"w4_support_matrices: {len(w4_support)}",
        f"w4_pass_matrices: {len(w4_pass)}",
        f"cfd_w4_support_matrices: {len(cfd_support)}",
        f"cfd_w4_pass_matrices: {len(cfd_pass)}",
        "",
        "Averages by width:",
    ]
    for width in WIDTHS:
        width_rows = [row for row in ok if int(row["w"]) == width]
        lines.append(
            f"- w={width}: matrices={len(width_rows)}, "
            f"pass={sum(1 for row in width_rows if row in passed)}, "
            f"mean_coverage={_mean([_float(row, 'covered_row_ratio') for row in width_rows]):.6g}, "
            f"median_coverage={_median([_float(row, 'covered_row_ratio') for row in width_rows]):.6g}, "
            f"max_scan_abs={max([_float(row, 'max_abs_error_scan_sim') for row in width_rows], default=0.0):.6g}, "
            f"max_seq_abs={max([_float(row, 'max_abs_error_seq') for row in width_rows], default=0.0):.6g}, "
            f"max_affine_abs={max([_float(row, 'max_abs_error_affine') for row in width_rows], default=0.0):.6g}"
        )

    if baseline_residual_warnings or baseline_inverse_warnings:
        lines.extend(["", "Baseline stability diagnostics:"])
        if baseline_residual_warnings:
            lines.append(
                "- High baseline residual: sequential SpTRSV itself does not recover"
                " `x_true` cleanly on these matrices, so validation there should be"
                " interpreted as equivalence-to-baseline, not proof of a well-conditioned solve."
            )
            for row in baseline_residual_warnings:
                lines.append(
                    f"  {row['matrix']} [w={row['w']}]: "
                    f"baseline_abs={_float(row, 'baseline_max_abs_error'):.6g}, "
                    f"baseline_rel_residual={_float(row, 'baseline_rel_residual'):.6g}"
                )
        if baseline_inverse_warnings:
            lines.append(
                "- Large baseline x-error but tiny residual: these matrices appear"
                " numerically ill-conditioned for the random right-hand side, even"
                " though the triangular solve remains algebraically consistent."
            )
            for row in baseline_inverse_warnings:
                lines.append(
                    f"  {row['matrix']} [w={row['w']}]: "
                    f"baseline_abs={_float(row, 'baseline_max_abs_error'):.6g}, "
                    f"baseline_rel_residual={_float(row, 'baseline_rel_residual'):.6g}"
                )

    lines.extend([
        "",
        "Per-matrix validation rows:",
    ])
    for row in ok:
        lines.append(
            f"- {row['matrix']} [{row['group']}], w={row['w']}: "
            f"verdict={row['verdict']}, regions={row['region_count']}, "
            f"validated={row['validated_region_count']}, failed={row['failed_region_count']}, "
            f"covered={_float(row, 'covered_row_ratio'):.6g}, "
            f"structural_ge8={_float(row, 'structural_ge8_row_coverage'):.6g}, "
            f"retention={_float(row, 'coverage_retention'):.6g}, "
            f"seq={_float(row, 'max_abs_error_seq'):.6g}, "
            f"affine={_float(row, 'max_abs_error_affine'):.6g}, "
            f"scan={_float(row, 'max_abs_error_scan_sim'):.6g}, "
            f"unresolved={row['external_unresolved_count']}"
        )

    failed = [row for row in rows if row.get("status") != "ok" or row.get("verdict", "").startswith("fail")]
    if failed:
        lines.extend(["", "Failure diagnostics:"])
        for row in failed:
            lines.append(
                f"- {row.get('matrix')} w={row.get('w')}: "
                f"verdict={row.get('verdict')}, error={row.get('error', '')}"
            )

    lines.extend([
        "",
        final_decision(rows),
    ])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args) -> list[dict]:
    factors = load_formal_factors(
        base.resolve_repo_relative_path(args.spd_manifest),
        base.resolve_repo_relative_path(args.cfd_manifest),
    )
    rows = []
    for formal_factor in factors:
        print(f"[cpu-validate] {formal_factor.formal_group}/{formal_factor.matrix_name}", flush=True)
        try:
            rows.extend(validate_factor(formal_factor))
        except Exception as exc:
            print(f"[WARN] {formal_factor.matrix_name} failed: {exc}", flush=True)
            for width in WIDTHS:
                rows.append(failed_width_row(formal_factor, width, exc))

    output_path = base.resolve_repo_relative_path(args.out)
    summary_path = base.resolve_repo_relative_path(args.summary_out)
    write_csv(rows, output_path)
    write_summary(summary_path, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CPU validation for formal bounded-width frontier regions."
    )
    parser.add_argument(
        "--spd-manifest",
        default=str(DEFAULT_SPD_MANIFEST),
        help=f"Formal SPD manifest (default: {DEFAULT_SPD_MANIFEST})",
    )
    parser.add_argument(
        "--cfd-manifest",
        default=str(DEFAULT_CFD_MANIFEST),
        help=f"Formal CFD manifest (default: {DEFAULT_CFD_MANIFEST})",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output CSV path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--summary-out",
        default=str(DEFAULT_SUMMARY),
        help=f"Summary path (default: {DEFAULT_SUMMARY})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_experiment(args)
    print(f"Done. Rows: {len(rows)}")
    print(f"CSV: {base.resolve_repo_relative_path(args.out)}")
    print(f"Summary: {base.resolve_repo_relative_path(args.summary_out)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU executor and cost-model study for formal bounded-width frontier regions.

This stage follows the structural and CPU-equivalence validation work:

- strict single-predecessor scan chains already failed;
- bounded-width frontier-safe regions are structurally present on a subset;
- CPU validation already showed those regions are algebraically equivalent.

The goal here is narrower: estimate whether a future scan-region executor has a
credible execution-cost story before any GPU prototype is attempted.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
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
from scan_sptrsv.formal_14_frontier_region_cpu_validation import (
    ERROR_TOL,
    NumericLowerFactor,
    lower_row_entries,
    read_numeric_lower_factor,
    solve_lower_sequential,
)


WIDTHS = (2, 4, 8)
MIN_EFFECTIVE_REGION_LEN = 8
DEFAULT_OUT = (
    base.RESULTS_DIR
    / "drop_experiments"
    / "formal_14_frontier_region_cpu_executor_study.csv"
)
DEFAULT_SUMMARY = (
    base.RESULTS_DIR
    / "drop_experiments"
    / "formal_14_frontier_region_cpu_executor_study_summary.txt"
)
DEFAULT_FIG_DIR = (
    base.REPO_ROOT
    / "figures"
    / "drop_experiments"
    / "formal_14_frontier_region_cpu_executor_study"
)


CSV_FIELDS = [
    "status",
    "error",
    "matrix",
    "group",
    "input_path",
    "w",
    "n",
    "nnz_L",
    "strict_lower_nnz",
    "critical_path_len",
    "region_count",
    "covered_rows",
    "covered_row_ratio",
    "ge8_covered_rows",
    "ge8_covered_ratio",
    "ge16_covered_rows",
    "ge16_covered_ratio",
    "baseline_max_abs_error",
    "baseline_residual",
    "region_seq_max_abs_error",
    "region_affine_max_abs_error",
    "region_scan_sim_max_abs_error",
    "region_scan_sim_max_rel_error",
    "correctness_pass",
    "external_unresolved_count",
    "external_nnz_total",
    "internal_nnz_total",
    "external_nnz_per_covered_row",
    "internal_nnz_per_covered_row",
    "avg_region_len",
    "max_region_len",
    "avg_level_span",
    "max_level_span",
    "avg_width",
    "max_width",
    "inspector_build_ms",
    "baseline_cpu_ms",
    "region_seq_replay_ms",
    "region_affine_ms",
    "scan_sim_ms",
    "inspector_reuse_assumed",
    "amortized_apply_ms",
    "baseline_est_ops",
    "external_reduction_est_ops",
    "affine_transition_est_ops",
    "scan_compose_est_ops",
    "metadata_est_bytes",
    "x_read_est_bytes",
    "x_write_est_bytes",
    "estimated_parallel_depth",
    "estimated_work_over_baseline",
    "estimated_depth_reduction",
    "estimated_profit_score",
    "verdict",
]


@dataclass(frozen=True)
class PlannedRegionRow:
    row: int
    diag: float
    external_entries: tuple[tuple[int, float], ...]
    internal_entries: tuple[tuple[int, float], ...]
    coeff_indices: tuple[int, ...]
    coeff_values: tuple[float, ...]
    coeff_dense: np.ndarray
    carry_pairs: tuple[tuple[int, int], ...]
    emit_pos: int
    state_before_width: int
    state_after_width: int


@dataclass(frozen=True)
class PlannedRegion:
    start: int
    end: int
    length: int
    level_span: int
    max_live_width: int
    external_nnz: int
    internal_nnz: int
    rows: tuple[PlannedRegionRow, ...]


@dataclass(frozen=True)
class StudyRowResult:
    x: np.ndarray
    max_abs_error: float
    max_rel_error: float
    elapsed_ms: float


def _float(row: dict, field: str) -> float:
    value = row.get(field, "")
    if value == "":
        return 0.0
    return float(value)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def matrix_seed(matrix_name: str) -> int:
    return zlib.crc32(matrix_name.encode("utf-8")) & 0xFFFFFFFF


def build_baseline_reference(formal_factor, numeric: NumericLowerFactor):
    rng = np.random.default_rng(matrix_seed(formal_factor.matrix_name))
    x_true = rng.standard_normal(numeric.n)
    b = numeric.csr @ x_true
    start = time.perf_counter()
    x_ref = solve_lower_sequential(numeric, b)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    baseline_abs = float(np.max(np.abs(x_ref - x_true))) if numeric.n else 0.0
    residual = numeric.csr @ x_ref - b
    residual_rel = float(
        np.linalg.norm(residual) / max(np.linalg.norm(b), 1e-300)
    )
    return x_true, b, x_ref, baseline_abs, residual_rel, elapsed_ms


def effective_regions_for_width(
    graph: base.StrictLowerDependencyGraph,
    levels: np.ndarray,
    width: int,
) -> tuple[tuple[Region, ...], tuple[Region, ...]]:
    all_regions = greedy_bounded_width_regions(graph, levels, width)
    effective = tuple(
        region for region in all_regions if region.length >= MIN_EFFECTIVE_REGION_LEN
    )
    ge16 = tuple(region for region in effective if region.length >= 16)
    return effective, ge16


def build_region_plan(
    factor: NumericLowerFactor,
    region: Region,
    width: int,
) -> tuple[PlannedRegion, int]:
    last_successor: dict[int, int] = {}
    unresolved = 0
    external_nnz = 0
    internal_nnz = 0
    for row in range(region.start, region.end + 1):
        entries, _ = lower_row_entries(factor, row)
        for pred, _ in entries:
            if pred < region.start:
                external_nnz += 1
            elif region.start <= pred < row:
                internal_nnz += 1
                last_successor[pred] = row
            else:
                unresolved += 1

    if unresolved:
        raise RuntimeError(
            f"Region [{region.start}, {region.end}] has {unresolved} unresolved deps"
        )

    state_rows: list[int] = []
    planned_rows: list[PlannedRegionRow] = []
    for row in range(region.start, region.end + 1):
        entries, diag = lower_row_entries(factor, row)
        state_pos = {state_row: pos for pos, state_row in enumerate(state_rows)}
        external_entries: list[tuple[int, float]] = []
        internal_entries: list[tuple[int, float]] = []
        coeff_indices: list[int] = []
        coeff_values: list[float] = []

        for pred, value in entries:
            if pred < region.start:
                external_entries.append((pred, value))
            elif region.start <= pred < row:
                if pred not in state_pos:
                    raise RuntimeError(
                        f"Missing live predecessor {pred} before row {row}"
                    )
                internal_entries.append((pred, value))
                coeff_indices.append(state_pos[pred])
                coeff_values.append(-value / diag)
            else:
                raise RuntimeError(f"Unresolved dependency {pred} for row {row}")

        next_rows = [
            state_row
            for state_row in state_rows
            if last_successor.get(state_row, -1) > row
        ]
        if last_successor.get(row, -1) > row:
            next_rows.append(row)
        next_rows.sort()
        if len(next_rows) > width:
            raise RuntimeError(
                f"Live width exceeded in region [{region.start}, {region.end}]"
            )

        carry_pairs: list[tuple[int, int]] = []
        emit_pos = -1
        for next_pos, state_row in enumerate(next_rows):
            if state_row == row:
                emit_pos = next_pos
            else:
                carry_pairs.append((next_pos, state_pos[state_row]))

        coeff_dense = np.zeros(width, dtype=np.float64)
        for coeff_index, coeff_value in zip(coeff_indices, coeff_values, strict=False):
            coeff_dense[coeff_index] = coeff_value

        planned_rows.append(
            PlannedRegionRow(
                row=row,
                diag=float(diag),
                external_entries=tuple(external_entries),
                internal_entries=tuple(internal_entries),
                coeff_indices=tuple(coeff_indices),
                coeff_values=tuple(coeff_values),
                coeff_dense=coeff_dense,
                carry_pairs=tuple(carry_pairs),
                emit_pos=emit_pos,
                state_before_width=len(state_rows),
                state_after_width=len(next_rows),
            )
        )
        state_rows = next_rows

    return (
        PlannedRegion(
            start=region.start,
            end=region.end,
            length=region.length,
            level_span=region.level_span,
            max_live_width=region.max_live_width,
            external_nnz=external_nnz,
            internal_nnz=internal_nnz,
            rows=tuple(planned_rows),
        ),
        unresolved,
    )


def build_region_plans(
    factor: NumericLowerFactor,
    regions: tuple[Region, ...],
    width: int,
) -> tuple[tuple[PlannedRegion, ...], int]:
    plans: list[PlannedRegion] = []
    unresolved_total = 0
    for region in regions:
        plan, unresolved = build_region_plan(factor, region, width)
        plans.append(plan)
        unresolved_total += unresolved
    return tuple(plans), unresolved_total


def _solve_single_row(
    factor: NumericLowerFactor,
    b: np.ndarray,
    x: np.ndarray,
    row: int,
) -> float:
    entries, diag = lower_row_entries(factor, row)
    lower_sum = 0.0
    for pred, value in entries:
        lower_sum += value * x[pred]
    return (float(b[row]) - lower_sum) / diag


def run_region_seq_replay(
    factor: NumericLowerFactor,
    b: np.ndarray,
    plans: tuple[PlannedRegion, ...],
) -> np.ndarray:
    plan_by_start = {plan.start: plan for plan in plans}
    x = np.zeros(factor.n, dtype=np.float64)
    row = 0
    while row < factor.n:
        plan = plan_by_start.get(row)
        if plan is None:
            x[row] = _solve_single_row(factor, b, x, row)
            row += 1
            continue

        local_x: dict[int, float] = {}
        for row_plan in plan.rows:
            rhs = float(b[row_plan.row])
            for pred, value in row_plan.external_entries:
                rhs -= value * x[pred]
            for pred, value in row_plan.internal_entries:
                rhs -= value * local_x[pred]
            x_value = rhs / row_plan.diag
            local_x[row_plan.row] = x_value
            x[row_plan.row] = x_value
        row = plan.end + 1
    return x


def run_region_affine_executor(
    factor: NumericLowerFactor,
    b: np.ndarray,
    plans: tuple[PlannedRegion, ...],
    width: int,
) -> np.ndarray:
    plan_by_start = {plan.start: plan for plan in plans}
    x = np.zeros(factor.n, dtype=np.float64)
    row = 0
    while row < factor.n:
        plan = plan_by_start.get(row)
        if plan is None:
            x[row] = _solve_single_row(factor, b, x, row)
            row += 1
            continue

        state = np.zeros(width, dtype=np.float64)
        for row_plan in plan.rows:
            rhs = float(b[row_plan.row])
            for pred, value in row_plan.external_entries:
                rhs -= value * x[pred]
            q = rhs / row_plan.diag
            x_value = float(row_plan.coeff_dense @ state + q)
            x[row_plan.row] = x_value
            next_state = np.zeros(width, dtype=np.float64)
            for next_pos, old_pos in row_plan.carry_pairs:
                next_state[next_pos] = state[old_pos]
            if row_plan.emit_pos >= 0:
                next_state[row_plan.emit_pos] = x_value
            state = next_state
        row = plan.end + 1
    return x


def run_region_scan_sim_executor(
    factor: NumericLowerFactor,
    b: np.ndarray,
    plans: tuple[PlannedRegion, ...],
    width: int,
) -> np.ndarray:
    plan_by_start = {plan.start: plan for plan in plans}
    x = np.zeros(factor.n, dtype=np.float64)
    row = 0
    while row < factor.n:
        plan = plan_by_start.get(row)
        if plan is None:
            x[row] = _solve_single_row(factor, b, x, row)
            row += 1
            continue

        transitions = []
        for row_plan in plan.rows:
            rhs = float(b[row_plan.row])
            for pred, value in row_plan.external_entries:
                rhs -= value * x[pred]
            q = rhs / row_plan.diag
            transition_a = np.zeros((width, width), dtype=np.float64)
            transition_c = np.zeros(width, dtype=np.float64)
            for next_pos, old_pos in row_plan.carry_pairs:
                transition_a[next_pos, old_pos] = 1.0
            if row_plan.emit_pos >= 0:
                transition_a[row_plan.emit_pos, :] = row_plan.coeff_dense
                transition_c[row_plan.emit_pos] = q
            transitions.append(
                (
                    transition_a,
                    transition_c,
                    row_plan.coeff_dense,
                    q,
                    row_plan.row,
                )
            )

        prefix_a = np.eye(width, dtype=np.float64)
        prefix_c = np.zeros(width, dtype=np.float64)
        for transition_a, transition_c, coeff_dense, q, plan_row in transitions:
            state_before = prefix_c
            x[plan_row] = float(coeff_dense @ state_before + q)
            prefix_a, prefix_c = (
                transition_a @ prefix_a,
                transition_a @ prefix_c + transition_c,
            )
        row = plan.end + 1
    return x


def timed_executor(
    fn,
    x_ref: np.ndarray,
):
    start = time.perf_counter()
    x = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    abs_error = np.abs(x - x_ref)
    max_abs = float(np.max(abs_error)) if x.size else 0.0
    rel_error = abs_error / np.maximum(np.abs(x_ref), 1e-300)
    max_rel = float(np.max(rel_error)) if x.size else 0.0
    return StudyRowResult(
        x=x,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        elapsed_ms=elapsed_ms,
    )


def compose_cost(width: int) -> float:
    return float(2 * width**3 + 2 * width**2)


def estimate_parallel_depth(n_rows: int, plans: tuple[PlannedRegion, ...]) -> float:
    covered_rows = sum(plan.length for plan in plans)
    uncovered_rows = n_rows - covered_rows
    depth = float(uncovered_rows)
    for plan in plans:
        depth += float(math.ceil(math.log2(plan.length)) + 1)
    return depth


def row_verdict(row: dict) -> str:
    if row.get("status") != "ok":
        return "failed"
    if row.get("correctness_pass") != "true":
        return "fail-correctness"
    if (
        _float(row, "covered_row_ratio") >= 0.10
        and _float(row, "estimated_depth_reduction") >= 0.20
        and _float(row, "estimated_work_over_baseline") <= 2.0
        and _float(row, "estimated_profit_score") > 0.0
    ):
        return "support"
    if (
        _float(row, "covered_row_ratio") >= 0.10
        and _float(row, "estimated_depth_reduction") >= 0.10
        and _float(row, "estimated_work_over_baseline") <= 3.0
    ):
        return "marginal"
    return "not-support"


def analyze_factor_width(formal_factor, width: int) -> dict:
    numeric = read_numeric_lower_factor(formal_factor.factor_l_path)
    graph = base.read_strict_lower_dependency_graph_mtx(formal_factor.factor_l_path)
    levels, _, _ = base.build_frontier_safe_parent(graph)
    critical_path_len = int(np.max(levels)) if graph.n_rows else 0
    _, b, x_ref, baseline_abs_error, baseline_residual, baseline_ms = (
        build_baseline_reference(formal_factor, numeric)
    )

    regions, ge16_regions = effective_regions_for_width(graph, levels, width)
    inspector_start = time.perf_counter()
    plans, unresolved_total = build_region_plans(numeric, regions, width)
    inspector_build_ms = (time.perf_counter() - inspector_start) * 1000.0

    summary = summarize_regions(
        regions,
        levels,
        graph.n_rows,
        critical_path_len,
    )
    ge16_rows = sum(region.length for region in ge16_regions)
    covered_rows = sum(plan.length for plan in plans)
    external_nnz_total = sum(plan.external_nnz for plan in plans)
    internal_nnz_total = sum(plan.internal_nnz for plan in plans)
    region_lengths = [plan.length for plan in plans]
    level_spans = [plan.level_span for plan in plans]
    width_samples = [
        max(row_plan.state_before_width, row_plan.state_after_width)
        for plan in plans
        for row_plan in plan.rows
    ]
    covered_rows_nonzero = covered_rows if covered_rows else 0

    seq_result = timed_executor(
        lambda: run_region_seq_replay(numeric, b, plans),
        x_ref,
    )
    affine_result = timed_executor(
        lambda: run_region_affine_executor(numeric, b, plans, width),
        x_ref,
    )
    scan_result = timed_executor(
        lambda: run_region_scan_sim_executor(numeric, b, plans, width),
        x_ref,
    )

    tolerance = max(ERROR_TOL, 100.0 * baseline_abs_error)
    correctness_pass = (
        unresolved_total == 0
        and seq_result.max_abs_error <= tolerance
        and affine_result.max_abs_error <= tolerance
        and scan_result.max_abs_error <= tolerance
    )

    baseline_est_ops = float(graph.strict_lower_nnz + graph.n_rows)
    covered_strict_nnz = float(external_nnz_total + internal_nnz_total)
    uncovered_rows = float(graph.n_rows - covered_rows)
    uncovered_strict_nnz = float(max(graph.strict_lower_nnz - covered_strict_nnz, 0.0))
    uncovered_est_ops = uncovered_strict_nnz + uncovered_rows
    external_reduction_est_ops = float(external_nnz_total)
    affine_transition_est_ops = float(
        sum(
            (2 * len(row_plan.coeff_indices) + 1)
            for plan in plans
            for row_plan in plan.rows
        )
    )
    scan_compose_est_ops = float(
        sum(
            2 * max(plan.length - 1, 0) * compose_cost(width)
            for plan in plans
        )
    )
    hybrid_work_est = (
        uncovered_est_ops
        + external_reduction_est_ops
        + affine_transition_est_ops
        + scan_compose_est_ops
    )
    metadata_est_bytes = float(
        external_nnz_total * 12
        + internal_nnz_total * 12
        + covered_rows * (8 * (width * width + 2 * width) + 16)
    )
    x_read_est_bytes = float((uncovered_strict_nnz + external_nnz_total) * 8)
    x_write_est_bytes = float(graph.n_rows * 8)
    estimated_parallel_depth = estimate_parallel_depth(graph.n_rows, plans)
    estimated_depth_reduction = (
        max(graph.n_rows - estimated_parallel_depth, 0.0) / graph.n_rows
        if graph.n_rows
        else 0.0
    )
    estimated_work_over_baseline = (
        hybrid_work_est / baseline_est_ops if baseline_est_ops else 0.0
    )
    estimated_profit_score = (
        (covered_rows / graph.n_rows) * estimated_depth_reduction
        - max(estimated_work_over_baseline - 1.0, 0.0)
        if graph.n_rows
        else 0.0
    )

    row = {
        "status": "ok",
        "error": "",
        "matrix": formal_factor.matrix_name,
        "group": formal_factor.formal_group,
        "input_path": str(formal_factor.factor_l_path),
        "w": width,
        "n": graph.n_rows,
        "nnz_L": numeric.nnz,
        "strict_lower_nnz": graph.strict_lower_nnz,
        "critical_path_len": critical_path_len,
        "region_count": len(plans),
        "covered_rows": covered_rows,
        "covered_row_ratio": (covered_rows / graph.n_rows) if graph.n_rows else 0.0,
        "ge8_covered_rows": covered_rows,
        "ge8_covered_ratio": (covered_rows / graph.n_rows) if graph.n_rows else 0.0,
        "ge16_covered_rows": ge16_rows,
        "ge16_covered_ratio": (ge16_rows / graph.n_rows) if graph.n_rows else 0.0,
        "baseline_max_abs_error": baseline_abs_error,
        "baseline_residual": baseline_residual,
        "region_seq_max_abs_error": seq_result.max_abs_error,
        "region_affine_max_abs_error": affine_result.max_abs_error,
        "region_scan_sim_max_abs_error": scan_result.max_abs_error,
        "region_scan_sim_max_rel_error": scan_result.max_rel_error,
        "correctness_pass": "true" if correctness_pass else "false",
        "external_unresolved_count": unresolved_total,
        "external_nnz_total": external_nnz_total,
        "internal_nnz_total": internal_nnz_total,
        "external_nnz_per_covered_row": (
            external_nnz_total / covered_rows_nonzero if covered_rows_nonzero else 0.0
        ),
        "internal_nnz_per_covered_row": (
            internal_nnz_total / covered_rows_nonzero if covered_rows_nonzero else 0.0
        ),
        "avg_region_len": _mean(region_lengths),
        "max_region_len": max(region_lengths, default=0),
        "avg_level_span": _mean(level_spans),
        "max_level_span": max(level_spans, default=0),
        "avg_width": _mean(width_samples),
        "max_width": max(width_samples, default=0),
        "inspector_build_ms": inspector_build_ms,
        "baseline_cpu_ms": baseline_ms,
        "region_seq_replay_ms": seq_result.elapsed_ms,
        "region_affine_ms": affine_result.elapsed_ms,
        "scan_sim_ms": scan_result.elapsed_ms,
        "inspector_reuse_assumed": "true",
        "amortized_apply_ms": affine_result.elapsed_ms,
        "baseline_est_ops": baseline_est_ops,
        "external_reduction_est_ops": external_reduction_est_ops,
        "affine_transition_est_ops": affine_transition_est_ops,
        "scan_compose_est_ops": scan_compose_est_ops,
        "metadata_est_bytes": metadata_est_bytes,
        "x_read_est_bytes": x_read_est_bytes,
        "x_write_est_bytes": x_write_est_bytes,
        "estimated_parallel_depth": estimated_parallel_depth,
        "estimated_work_over_baseline": estimated_work_over_baseline,
        "estimated_depth_reduction": estimated_depth_reduction,
        "estimated_profit_score": estimated_profit_score,
    }
    row["verdict"] = row_verdict(row)
    return row


def failed_row(formal_factor, width: int, exc: Exception) -> dict:
    return {
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "matrix": formal_factor.matrix_name,
        "group": formal_factor.formal_group,
        "input_path": str(formal_factor.factor_l_path),
        "w": width,
        "correctness_pass": "false",
        "inspector_reuse_assumed": "true",
        "verdict": "failed",
    }


def ok_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("status") == "ok"]


def passed_rows(rows: list[dict]) -> list[dict]:
    return [
        row for row in ok_rows(rows)
        if row.get("correctness_pass") == "true"
    ]


def support_rows(rows: list[dict]) -> list[dict]:
    return [row for row in ok_rows(rows) if row.get("verdict") == "support"]


def marginal_rows(rows: list[dict]) -> list[dict]:
    return [row for row in ok_rows(rows) if row.get("verdict") == "marginal"]


def best_rows_by_matrix(rows: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    by_matrix: dict[str, list[dict]] = {}
    for row in ok_rows(rows):
        by_matrix.setdefault(row["matrix"], []).append(row)
    for matrix, matrix_rows in by_matrix.items():
        best[matrix] = max(
            matrix_rows,
            key=lambda row: (
                row.get("correctness_pass") == "true",
                _float(row, "estimated_profit_score"),
                _float(row, "covered_row_ratio"),
                -_float(row, "estimated_work_over_baseline"),
            ),
        )
    return best


def final_decision(rows: list[dict]) -> str:
    supported = support_rows(rows)
    cfd_supported = [row for row in supported if row.get("group") == "CFD"]
    spd_supported = [row for row in supported if row.get("group") == "SPD"]
    if len(supported) >= 4 or len(cfd_supported) >= 3:
        if len(spd_supported) == 0 and len(cfd_supported) >= 3:
            return (
                "Final decision: B. Structurally and numerically valid, but cost is "
                "only compelling in a narrow CFD / nonsymmetric PDE scope."
            )
        return "Final decision: A. Support bounded-width scan-region executor prototype."
    best = best_rows_by_matrix(rows)
    best_marginal = [row for row in best.values() if row.get("verdict") == "marginal"]
    best_cfd_marginal = [row for row in best_marginal if row.get("group") == "CFD"]
    if best_cfd_marginal and not spd_supported:
        return (
            "Final decision: B. Structurally and numerically valid, but cost is "
            "marginal; do not start a GPU prototype yet, and only keep a narrow "
            "CFD-focused follow-up if needed."
        )
    return "Final decision: C. Not support scan-region execution; abandon scan direction."


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_summary(summary_path: Path, rows: list[dict]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    ok = ok_rows(rows)
    passed = passed_rows(rows)
    supported = support_rows(rows)
    marginal = marginal_rows(rows)
    best = best_rows_by_matrix(rows)
    best_supported = [row for row in best.values() if row.get("verdict") == "support"]
    best_cfd_supported = [
        row for row in best_supported if row.get("group") == "CFD"
    ]
    best_spd_supported = [
        row for row in best_supported if row.get("group") == "SPD"
    ]

    baseline_residual_warnings = [
        row for row in ok if _float(row, "baseline_residual") > 1e-8
    ]
    baseline_x_error_warnings = [
        row
        for row in ok
        if _float(row, "baseline_residual") <= 1e-8
        and _float(row, "baseline_max_abs_error") > 1e-6
    ]

    lines = [
        "Formal 14 frontier-region CPU executor study summary",
        "",
        "Scope:",
        "- CPU study only. No dropping, no new matrices, no factor regeneration, no GPU kernel.",
        "- Uses effective frontier-safe bounded-width regions with region_len >= 8.",
        "- Compares baseline sequential SpTRSV, region sequential replay, affine-state executor, and scan-style prefix-compose simulation.",
        "",
        "Cost model notes:",
        "- baseline_est_ops uses strict-lower edge touches plus one solve step per row.",
        "- external_reduction_est_ops counts covered-row external dependencies.",
        "- affine_transition_est_ops counts sparse small-state row-output arithmetic only.",
        "- scan_compose_est_ops uses a conservative upsweep+downsweep estimate: 2*(L-1)*compose_cost(w) per region.",
        "- estimated_parallel_depth uses a row-order baseline depth n, replacing each region length L by ceil(log2(L)) + 1.",
        "- estimated_profit_score = covered_row_ratio * estimated_depth_reduction - max(estimated_work_over_baseline - 1, 0).",
        "",
        "Decision thresholds for support rows:",
        "- correctness_pass = true",
        "- covered_row_ratio >= 0.10",
        "- estimated_depth_reduction >= 0.20",
        "- estimated_work_over_baseline <= 2.0",
        "- estimated_profit_score > 0",
        "",
        f"rows: {len(rows)}",
        f"ok_rows: {len(ok)}",
        f"correctness_pass_rows: {len(passed)}",
        f"support_rows: {len(supported)}",
        f"marginal_rows: {len(marginal)}",
        f"best_support_matrices: {len(best_supported)}",
        f"best_support_spd_matrices: {len(best_spd_supported)}",
        f"best_support_cfd_matrices: {len(best_cfd_supported)}",
        "",
        "Width aggregates:",
    ]

    for width in WIDTHS:
        width_rows = [row for row in ok if int(row["w"]) == width]
        lines.append(
            f"- w={width}: matrices={len(width_rows)}, "
            f"support={sum(1 for row in width_rows if row.get('verdict') == 'support')}, "
            f"marginal={sum(1 for row in width_rows if row.get('verdict') == 'marginal')}, "
            f"mean_coverage={_mean([_float(row, 'covered_row_ratio') for row in width_rows]):.6g}, "
            f"median_coverage={_median([_float(row, 'covered_row_ratio') for row in width_rows]):.6g}, "
            f"mean_work_over_baseline={_mean([_float(row, 'estimated_work_over_baseline') for row in width_rows]):.6g}, "
            f"mean_depth_reduction={_mean([_float(row, 'estimated_depth_reduction') for row in width_rows]):.6g}, "
            f"mean_profit={_mean([_float(row, 'estimated_profit_score') for row in width_rows]):.6g}"
        )

    lines.extend(["", "Best width per matrix:"])
    for matrix in sorted(best):
        row = best[matrix]
        lines.append(
            f"- {matrix} [{row['group']}]: best_w={row['w']}, verdict={row['verdict']}, "
            f"coverage={_float(row, 'covered_row_ratio'):.6g}, "
            f"ge16={_float(row, 'ge16_covered_ratio'):.6g}, "
            f"depth_reduction={_float(row, 'estimated_depth_reduction'):.6g}, "
            f"work_over_baseline={_float(row, 'estimated_work_over_baseline'):.6g}, "
            f"profit={_float(row, 'estimated_profit_score'):.6g}"
        )

    lines.extend([
        "",
        "Interpretation by group:",
    ])
    for group in ("SPD", "CFD"):
        group_best = [row for row in best.values() if row.get("group") == group]
        group_support = [row for row in group_best if row.get("verdict") == "support"]
        group_marginal = [row for row in group_best if row.get("verdict") == "marginal"]
        lines.append(
            f"- {group}: support_matrices={len(group_support)}, "
            f"marginal_matrices={len(group_marginal)}, "
            f"mean_best_coverage={_mean([_float(row, 'covered_row_ratio') for row in group_best]):.6g}, "
            f"mean_best_work={_mean([_float(row, 'estimated_work_over_baseline') for row in group_best]):.6g}, "
            f"mean_best_depth_reduction={_mean([_float(row, 'estimated_depth_reduction') for row in group_best]):.6g}"
        )

    if baseline_residual_warnings or baseline_x_error_warnings:
        lines.extend(["", "Baseline stability diagnostics:"])
        if baseline_residual_warnings:
            lines.append(
                "- High baseline residual: these rows should be interpreted as executor-to-baseline equivalence only."
            )
            for row in baseline_residual_warnings:
                lines.append(
                    f"  {row['matrix']} [w={row['w']}]: "
                    f"baseline_abs={_float(row, 'baseline_max_abs_error'):.6g}, "
                    f"baseline_residual={_float(row, 'baseline_residual'):.6g}"
                )
        if baseline_x_error_warnings:
            lines.append(
                "- Large baseline x-error but tiny residual: likely conditioning/scale effects rather than executor mismatch."
            )
            for row in baseline_x_error_warnings:
                lines.append(
                    f"  {row['matrix']} [w={row['w']}]: "
                    f"baseline_abs={_float(row, 'baseline_max_abs_error'):.6g}, "
                    f"baseline_residual={_float(row, 'baseline_residual'):.6g}"
                )

    lines.extend(["", "Support gating observations:"])
    if best_supported:
        lines.append(
            f"- {len(best_supported)} matrices clear the executor-support bar on their best width."
        )
    else:
        lines.append(
            "- No matrix clears the executor-support bar on its best width; the main blockers are low depth reduction at w=2 and high work inflation at w=4/w=8."
        )

    if best_spd_supported and best_cfd_supported:
        lines.append(
            "- Both SPD and CFD groups contain supported matrices, so the direction is not limited to a single family."
        )
    elif best_cfd_supported and not best_spd_supported:
        lines.append(
            "- Only CFD matrices clear the support bar; this looks like a narrow GMRES/ILU opportunity, not a general SpTRSV path."
        )
    elif best_spd_supported and not best_cfd_supported:
        lines.append(
            "- Only SPD matrices clear the support bar; this is unusual and should be treated cautiously."
        )

    lines.extend([
        "",
        "Per-row details:",
    ])
    for row in ok:
        lines.append(
            f"- {row['matrix']} [{row['group']}], w={row['w']}: verdict={row['verdict']}, "
            f"pass={row['correctness_pass']}, covered={_float(row, 'covered_row_ratio'):.6g}, "
            f"ge16={_float(row, 'ge16_covered_ratio'):.6g}, ext_per_row={_float(row, 'external_nnz_per_covered_row'):.6g}, "
            f"int_per_row={_float(row, 'internal_nnz_per_covered_row'):.6g}, "
            f"work={_float(row, 'estimated_work_over_baseline'):.6g}, "
            f"depth_reduction={_float(row, 'estimated_depth_reduction'):.6g}, "
            f"profit={_float(row, 'estimated_profit_score'):.6g}, "
            f"baseline_ms={_float(row, 'baseline_cpu_ms'):.6g}, "
            f"affine_ms={_float(row, 'region_affine_ms'):.6g}, "
            f"scan_ms={_float(row, 'scan_sim_ms'):.6g}"
        )

    failed = [row for row in rows if row.get("status") != "ok"]
    if failed:
        lines.extend(["", "Failure diagnostics:"])
        for row in failed:
            lines.append(
                f"- {row.get('matrix')} w={row.get('w')}: {row.get('error', '')}"
            )

    lines.extend(["", final_decision(rows)])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(fig_dir: Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed; skip plots")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    ok = ok_rows(rows)
    labels = [f"{row['matrix']}\n{row['group']}" for row in ok]
    x = np.arange(len(labels))
    bar_width = 0.25

    def save_metric(metric: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(max(10.0, 0.55 * len(labels)), 5.2))
        for offset, width in enumerate(WIDTHS):
            width_rows = [row for row in ok if int(row["w"]) == width]
            width_map = {row["matrix"]: row for row in width_rows}
            values = [
                _float(width_map.get(row["matrix"], {}), metric)
                for row in ok
                if int(row["w"]) == WIDTHS[0]
            ]
            ax.bar(
                x + (offset - 1) * bar_width,
                values,
                width=bar_width,
                label=f"w={width}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=180)
        plt.close(fig)

    ordered_base = [row for row in ok if int(row["w"]) == WIDTHS[0]]
    ordered_labels = [f"{row['matrix']}\n{row['group']}" for row in ordered_base]
    x_base = np.arange(len(ordered_labels))

    def save_ordered_metric(metric: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(max(10.0, 0.55 * len(ordered_labels)), 5.2))
        for offset, width in enumerate(WIDTHS):
            width_rows = {row["matrix"]: row for row in ok if int(row["w"]) == width}
            values = [
                _float(width_rows[row["matrix"]], metric)
                for row in ordered_base
            ]
            ax.bar(
                x_base + (offset - 1) * bar_width,
                values,
                width=bar_width,
                label=f"w={width}",
            )
        ax.set_xticks(x_base)
        ax.set_xticklabels(ordered_labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=180)
        plt.close(fig)

    save_ordered_metric(
        "covered_row_ratio",
        "covered row ratio",
        "covered_row_ratio_by_matrix.png",
    )
    save_ordered_metric(
        "estimated_depth_reduction",
        "estimated depth reduction",
        "depth_reduction_by_matrix.png",
    )
    save_ordered_metric(
        "estimated_profit_score",
        "estimated profit score",
        "profit_score_by_matrix.png",
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    colors = {"SPD": "#1f77b4", "CFD": "#d62728"}
    for row in ok:
        ax.scatter(
            _float(row, "estimated_work_over_baseline"),
            _float(row, "estimated_depth_reduction"),
            color=colors.get(row["group"], "#444444"),
            marker={2: "o", 4: "s", 8: "^"}[int(row["w"])],
            s=54,
            alpha=0.85,
        )
        ax.text(
            _float(row, "estimated_work_over_baseline") + 0.01,
            _float(row, "estimated_depth_reduction") + 0.002,
            f"{row['matrix']}-w{row['w']}",
            fontsize=7,
        )
    ax.axvline(2.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.axhline(0.20, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xlabel("estimated work over baseline")
    ax.set_ylabel("estimated depth reduction")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "work_vs_depth_scatter.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x_group = np.arange(len(WIDTHS))
    width_bar = 0.35
    for group_index, group in enumerate(("SPD", "CFD")):
        values = []
        for width in WIDTHS:
            group_values = [
                _float(row, "estimated_profit_score")
                for row in ok
                if row["group"] == group and int(row["w"]) == width
            ]
            values.append(_mean(group_values))
        ax.bar(
            x_group + (group_index - 0.5) * width_bar,
            values,
            width=width_bar,
            label=group,
        )
    ax.set_xticks(x_group)
    ax.set_xticklabels([f"w={width}" for width in WIDTHS])
    ax.set_ylabel("mean profit score")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "group_mean_profit_by_width.png", dpi=180)
    plt.close(fig)


def run_experiment(args) -> list[dict]:
    factors = load_formal_factors(
        base.resolve_repo_relative_path(args.spd_manifest),
        base.resolve_repo_relative_path(args.cfd_manifest),
    )
    rows: list[dict] = []
    for factor in factors:
        for width in WIDTHS:
            print(
                f"[executor-study] {factor.formal_group}/{factor.matrix_name}, w={width}",
                flush=True,
            )
            try:
                rows.append(analyze_factor_width(factor, width))
            except Exception as exc:
                print(
                    f"[WARN] {factor.matrix_name}, w={width} failed: {exc}",
                    flush=True,
                )
                rows.append(failed_row(factor, width, exc))

    output_path = base.resolve_repo_relative_path(args.out)
    summary_path = base.resolve_repo_relative_path(args.summary_out)
    fig_dir = base.resolve_repo_relative_path(args.fig_dir)
    write_csv(rows, output_path)
    write_summary(summary_path, rows)
    if not args.no_plots:
        write_plots(fig_dir, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run formal 14 frontier-region CPU executor cost study."
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
    parser.add_argument(
        "--fig-dir",
        default=str(DEFAULT_FIG_DIR),
        help=f"Figure output directory (default: {DEFAULT_FIG_DIR})",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
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

#!/usr/bin/env python3
"""Formal 14-factor frontier/relaxed bounded-width region experiment."""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scan_sptrsv import analyze_scan_chains as base
from scan_sptrsv import drop_experiments as drop
from scan_sptrsv.formal_14_experiment import (
    DEFAULT_CFD_MANIFEST,
    DEFAULT_SPD_MANIFEST,
    FORMAL_CFD_NAMES,
    FORMAL_SPD_NAMES,
    FormalFactor,
    load_formal_factors,
)


WIDTHS = (1, 2, 4, 8)
DEFAULT_OUT = (
    base.RESULTS_DIR
    / "drop_experiments"
    / "formal_14_frontier_relaxed_region.csv"
)
DEFAULT_SUMMARY = (
    base.RESULTS_DIR
    / "drop_experiments"
    / "formal_14_frontier_relaxed_region_summary.txt"
)
DEFAULT_SPD_SUMMARY = (
    base.RESULTS_DIR
    / "drop_experiments"
    / "formal_14_frontier_relaxed_region_spd_summary.txt"
)
DEFAULT_CFD_SUMMARY = (
    base.RESULTS_DIR
    / "drop_experiments"
    / "formal_14_frontier_relaxed_region_cfd_summary.txt"
)
DEFAULT_FIG_DIR = (
    base.REPO_ROOT
    / "figures"
    / "drop_experiments"
    / "formal_14_frontier_relaxed_region"
)


BASE_FIELDS = [
    "status",
    "error",
    "matrix",
    "group",
    "input_path",
    "n",
    "nnz_L",
    "strict_lower_nnz",
    "multi_pred_rows",
    "multi_pred_ratio",
    "level_count",
    "critical_path_len",
    "strict_chain_coverage",
    "strict_chain_ge8_coverage",
    "strict_chain_ge16_coverage",
    "max_strict_chain_len",
]


def width_fields(width: int) -> list[str]:
    prefix = f"bw{width}"
    return [
        f"{prefix}_region_count",
        f"{prefix}_row_coverage",
        f"{prefix}_ge8_row_coverage",
        f"{prefix}_ge16_row_coverage",
        f"{prefix}_avg_region_len",
        f"{prefix}_max_region_len",
        f"{prefix}_avg_level_span",
        f"{prefix}_max_level_span",
        f"{prefix}_critical_path_coverage",
        f"{prefix}_external_nnz_per_row",
        f"{prefix}_internal_nnz_per_row",
    ]


CSV_FIELDS = BASE_FIELDS + [
    field
    for width in WIDTHS
    for field in width_fields(width)
]


@dataclass(frozen=True)
class Region:
    start: int
    end: int
    length: int
    max_live_width: int
    level_span: int
    internal_nnz: int
    external_nnz: int


@dataclass(frozen=True)
class WidthSummary:
    region_count: int
    row_coverage: float
    ge8_row_coverage: float
    ge16_row_coverage: float
    avg_region_len: float
    max_region_len: int
    avg_level_span: float
    max_level_span: int
    critical_path_coverage: float
    external_nnz_per_row: float
    internal_nnz_per_row: float
    regions: tuple[Region, ...]


class RangeAddMaxTree:
    """Segment tree for current-region live interval overlap."""

    def __init__(self, size: int):
        self.size = 1
        while self.size < max(size, 1):
            self.size <<= 1
        self.max_values = np.zeros(2 * self.size, dtype=np.int16)
        self.lazy = np.zeros(2 * self.size, dtype=np.int16)

    @property
    def max(self) -> int:
        return int(self.max_values[1])

    def add(self, left: int, right: int, value: int) -> None:
        if left > right:
            return
        self._add(left, right, value, 1, 0, self.size - 1)

    def _add(
        self,
        left: int,
        right: int,
        value: int,
        node: int,
        node_left: int,
        node_right: int,
    ) -> None:
        if left <= node_left and node_right <= right:
            self.max_values[node] += value
            self.lazy[node] += value
            return
        mid = (node_left + node_right) // 2
        if left <= mid:
            self._add(left, right, value, node * 2, node_left, mid)
        if right > mid:
            self._add(left, right, value, node * 2 + 1, mid + 1, node_right)
        self.max_values[node] = self.lazy[node] + max(
            self.max_values[node * 2],
            self.max_values[node * 2 + 1],
        )


def _float(row: dict, field: str) -> float:
    value = row.get(field, "")
    if value == "":
        return 0.0
    return float(value)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _row_internal_info(
    graph: base.StrictLowerDependencyGraph,
    row: int,
    region_start: int,
    width: int,
) -> tuple[bool, list[int], int, int]:
    """Return unique internal predecessor rows and edge counts for one row."""
    unique_internal = []
    seen_internal = set()
    internal_nnz = 0
    external_nnz = 0
    start = int(graph.indptr[row])
    end = int(graph.indptr[row + 1])
    for edge in range(start, end):
        pred = int(graph.indices[edge])
        if pred >= region_start:
            internal_nnz += 1
            if pred not in seen_internal:
                seen_internal.add(pred)
                unique_internal.append(pred)
                if len(unique_internal) > width:
                    return False, unique_internal, internal_nnz, external_nnz
        else:
            external_nnz += 1
    return True, unique_internal, internal_nnz, external_nnz


def greedy_bounded_width_regions(
    graph: base.StrictLowerDependencyGraph,
    levels: np.ndarray,
    width: int,
) -> tuple[Region, ...]:
    """
    Greedy contiguous region scan with a conservative live-frontier width.

    For a tentative region [s, e], each internal dependency p -> r creates a
    live interval over boundaries [p, r-1]. The live width is the maximum number
    of such distinct predecessor rows alive at any boundary. A row is accepted
    only if the max live width remains <= width.
    """
    n = graph.n_rows
    tree = RangeAddMaxTree(n)
    regions: list[Region] = []
    region_start = 0

    while region_start < n:
        region_end = region_start
        last_successor: dict[int, int] = {}
        applied_ranges: list[tuple[int, int]] = []
        level_min = int(levels[region_start])
        level_max = level_min
        internal_nnz_sum = 0
        external_nnz_sum = int(graph.pred_count[region_start])
        max_live_width = 0

        row = region_start + 1
        while row < n:
            ok, internal_preds, internal_nnz, external_nnz = _row_internal_info(
                graph,
                row,
                region_start,
                width,
            )
            if not ok:
                break

            changes: list[tuple[int, int, int, int | None]] = []
            for pred in internal_preds:
                old_last = last_successor.get(pred)
                interval_start = pred if old_last is None else old_last
                interval_end = row - 1
                if interval_start <= interval_end:
                    tree.add(interval_start, interval_end, 1)
                    changes.append((interval_start, interval_end, pred, old_last))

            if tree.max > width:
                for interval_start, interval_end, _, _ in reversed(changes):
                    tree.add(interval_start, interval_end, -1)
                break

            for interval_start, interval_end, pred, _ in changes:
                last_successor[pred] = row
                applied_ranges.append((interval_start, interval_end))

            region_end = row
            max_live_width = max(max_live_width, tree.max)
            internal_nnz_sum += internal_nnz
            external_nnz_sum += external_nnz
            row_level = int(levels[row])
            level_min = min(level_min, row_level)
            level_max = max(level_max, row_level)
            row += 1

        for interval_start, interval_end in reversed(applied_ranges):
            tree.add(interval_start, interval_end, -1)

        length = region_end - region_start + 1
        regions.append(
            Region(
                start=region_start,
                end=region_end,
                length=length,
                max_live_width=max_live_width,
                level_span=level_max - level_min + 1,
                internal_nnz=internal_nnz_sum,
                external_nnz=external_nnz_sum,
            )
        )
        region_start = region_end + 1

    return tuple(regions)


def summarize_regions(
    regions: tuple[Region, ...],
    levels: np.ndarray,
    n_rows: int,
    critical_path_len: int,
) -> WidthSummary:
    nontrivial = [region for region in regions if region.length >= 2]
    ge8 = [region for region in regions if region.length >= 8]
    ge16 = [region for region in regions if region.length >= 16]
    covered_rows = sum(region.length for region in nontrivial)
    ge8_rows = sum(region.length for region in ge8)
    ge16_rows = sum(region.length for region in ge16)
    level_spans = [region.level_span for region in nontrivial]
    effective_levels = set()
    for region in ge8:
        effective_levels.update(
            int(level) for level in levels[region.start : region.end + 1]
        )
    internal_nnz = sum(region.internal_nnz for region in nontrivial)
    external_nnz = sum(region.external_nnz for region in nontrivial)
    denominator = covered_rows if covered_rows else 0
    max_level_span = max(level_spans) if level_spans else 0

    return WidthSummary(
        region_count=len(nontrivial),
        row_coverage=(covered_rows / n_rows) if n_rows else 0.0,
        ge8_row_coverage=(ge8_rows / n_rows) if n_rows else 0.0,
        ge16_row_coverage=(ge16_rows / n_rows) if n_rows else 0.0,
        avg_region_len=(covered_rows / len(nontrivial)) if nontrivial else 0.0,
        max_region_len=max((region.length for region in nontrivial), default=0),
        avg_level_span=_mean(level_spans),
        max_level_span=max_level_span,
        critical_path_coverage=(
            len(effective_levels) / critical_path_len
            if critical_path_len
            else 0.0
        ),
        external_nnz_per_row=(external_nnz / denominator) if denominator else 0.0,
        internal_nnz_per_row=(internal_nnz / denominator) if denominator else 0.0,
        regions=regions,
    )


def analyze_factor(factor: FormalFactor) -> dict:
    graph = base.read_strict_lower_dependency_graph_mtx(factor.factor_l_path)
    levels, _, _ = base.build_frontier_safe_parent(graph)
    level_count = int(np.unique(levels).size) if graph.n_rows else 0
    critical_path_len = int(np.max(levels)) if graph.n_rows else 0
    strict = drop.strict_chain_summary(graph)
    n_rows = graph.n_rows

    row = {
        "status": "ok",
        "error": "",
        "matrix": factor.matrix_name,
        "group": factor.formal_group,
        "input_path": str(factor.factor_l_path),
        "n": n_rows,
        "nnz_L": graph.strict_lower_nnz + n_rows,
        "strict_lower_nnz": graph.strict_lower_nnz,
        "multi_pred_rows": int(np.count_nonzero(graph.pred_count > 1)),
        "multi_pred_ratio": (
            int(np.count_nonzero(graph.pred_count > 1)) / n_rows
            if n_rows
            else 0.0
        ),
        "level_count": level_count,
        "critical_path_len": critical_path_len,
        "strict_chain_coverage": (
            strict.row_coverage / n_rows if n_rows else 0.0
        ),
        "strict_chain_ge8_coverage": (
            strict.rows_ge[8] / n_rows if n_rows else 0.0
        ),
        "strict_chain_ge16_coverage": (
            strict.rows_ge[16] / n_rows if n_rows else 0.0
        ),
        "max_strict_chain_len": strict.max_length,
    }

    for width in WIDTHS:
        regions = greedy_bounded_width_regions(graph, levels, width)
        summary = summarize_regions(regions, levels, n_rows, critical_path_len)
        prefix = f"bw{width}"
        row.update({
            f"{prefix}_region_count": summary.region_count,
            f"{prefix}_row_coverage": summary.row_coverage,
            f"{prefix}_ge8_row_coverage": summary.ge8_row_coverage,
            f"{prefix}_ge16_row_coverage": summary.ge16_row_coverage,
            f"{prefix}_avg_region_len": summary.avg_region_len,
            f"{prefix}_max_region_len": summary.max_region_len,
            f"{prefix}_avg_level_span": summary.avg_level_span,
            f"{prefix}_max_level_span": summary.max_level_span,
            f"{prefix}_critical_path_coverage": summary.critical_path_coverage,
            f"{prefix}_external_nnz_per_row": summary.external_nnz_per_row,
            f"{prefix}_internal_nnz_per_row": summary.internal_nnz_per_row,
        })

    return row


def failed_row(factor: FormalFactor, exc: Exception) -> dict:
    return {
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "matrix": factor.matrix_name,
        "group": factor.formal_group,
        "input_path": str(factor.factor_l_path),
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _ok_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("status") == "ok"]


def support_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    ok_rows = _ok_rows(rows)
    counts: dict[str, dict[str, int]] = {}
    for width in WIDTHS:
        counts[f"bw{width}"] = {
            "ge8_ge_010": sum(
                1 for row in ok_rows
                if _float(row, f"bw{width}_ge8_row_coverage") >= 0.10
            ),
            "critical_ge_020": sum(
                1 for row in ok_rows
                if _float(row, f"bw{width}_critical_path_coverage") >= 0.20
            ),
        }
    return counts


def final_decision(rows: list[dict]) -> str:
    ok_rows = _ok_rows(rows)
    counts = support_counts(ok_rows)
    condition_all_ge8 = any(
        counts[f"bw{width}"]["ge8_ge_010"] >= 4
        for width in (1, 2, 4)
    )
    condition_all_critical = any(
        counts[f"bw{width}"]["critical_ge_020"] >= 4
        for width in (1, 2, 4)
    )
    cfd_rows = [row for row in ok_rows if row.get("group") == "CFD"]
    cfd_bw4_support = sum(
        1 for row in cfd_rows
        if (
            _float(row, "bw4_ge8_row_coverage") >= 0.10
            or _float(row, "bw4_critical_path_coverage") >= 0.20
        )
    )
    if condition_all_ge8 or condition_all_critical or cfd_bw4_support >= 3:
        return (
            "Final decision: B. Narrow support for scan-region direction; "
            "next step should be CPU numerical validation, not GPU kernel."
        )
    return "Final decision: C. Not support scan-chain / scan-region direction."


def _top_lines(rows: list[dict], field: str, limit: int = 5) -> list[str]:
    ordered = sorted(_ok_rows(rows), key=lambda row: _float(row, field), reverse=True)
    return [
        f"- {row['matrix']} [{row['group']}]: {field}={_float(row, field):.6g}"
        for row in ordered[:limit]
    ]


def write_summary(summary_path: Path, rows: list[dict], title: str, group: str | None = None) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    ok_rows = _ok_rows(rows)
    failed_rows = [row for row in rows if row.get("status") == "failed"]
    counts = support_counts(ok_rows)
    decision = final_decision(rows)
    if group is not None and group != "ALL":
        decision = group_decision(rows, group)

    lines = [
        title,
        "",
        "Scope:",
        "- Formal 14 manifest factors only.",
        "- No dropping, no new downloads, no factor regeneration, no GPU solver.",
        f"- group={group or 'ALL'}, matrices={len(ok_rows)}, failed={len(failed_rows)}",
        "",
        "Greedy bounded-width definition:",
        "- Regions are contiguous row-order intervals.",
        "- External predecessors are considered already resolved by the completed/frontier set.",
        "- Internal dependency p -> r creates a live interval across boundaries [p, r-1].",
        "- A bounded-width-w region is accepted only if max live internal rows <= w.",
        "- Coverage fields are ratios over n_rows; region_count ignores singleton regions.",
        "- critical_path_coverage is distinct dependency levels touched by length >= 8 regions divided by critical_path_len.",
        "",
        "Decision thresholds:",
        "- Continue only if w <= 4 has at least 4 / 14 matrices with ge8 row coverage >= 0.10.",
        "- Or w <= 4 has at least 4 / 14 matrices with critical path coverage >= 0.20.",
        "- Or CFD group has at least 3 / 6 matrices satisfying bw4 ge8 coverage >= 0.10 or bw4 critical path coverage >= 0.20.",
        "",
        "Support counts:",
    ]
    for width in WIDTHS:
        item = counts[f"bw{width}"]
        lines.append(
            f"- bw{width}: ge8>=0.10 matrices={item['ge8_ge_010']}, "
            f"critical_path>=0.20 matrices={item['critical_ge_020']}"
        )

    lines.extend([
        "",
        "Averages by width:",
    ])
    for width in WIDTHS:
        ge8_values = [
            _float(row, f"bw{width}_ge8_row_coverage") for row in ok_rows
        ]
        ge16_values = [
            _float(row, f"bw{width}_ge16_row_coverage") for row in ok_rows
        ]
        critical_values = [
            _float(row, f"bw{width}_critical_path_coverage") for row in ok_rows
        ]
        max_len_values = [
            _float(row, f"bw{width}_max_region_len") for row in ok_rows
        ]
        lines.append(
            f"- bw{width}: mean_ge8={_mean(ge8_values):.6g}, "
            f"median_ge8={_median(ge8_values):.6g}, "
            f"mean_ge16={_mean(ge16_values):.6g}, "
            f"mean_critical={_mean(critical_values):.6g}, "
            f"median_critical={_median(critical_values):.6g}, "
            f"max_region_len_max={max(max_len_values) if max_len_values else 0:.6g}"
        )

    lines.extend([
        "",
        "Top bw4 ge8 row coverage:",
        *_top_lines(ok_rows, "bw4_ge8_row_coverage"),
        "",
        "Top bw4 critical path coverage:",
        *_top_lines(ok_rows, "bw4_critical_path_coverage"),
        "",
        "Strict baseline:",
    ])
    strict_ge8 = [_float(row, "strict_chain_ge8_coverage") for row in ok_rows]
    strict_ge16 = [_float(row, "strict_chain_ge16_coverage") for row in ok_rows]
    strict_max = [_float(row, "max_strict_chain_len") for row in ok_rows]
    lines.append(
        f"- mean_strict_ge8={_mean(strict_ge8):.6g}, "
        f"mean_strict_ge16={_mean(strict_ge16):.6g}, "
        f"max_strict_chain_len={max(strict_max) if strict_max else 0:.6g}"
    )

    lines.extend([
        "",
        "Per-matrix core metrics:",
    ])
    for row in ok_rows:
        lines.append(
            f"- {row['matrix']} [{row['group']}]: "
            f"multi_pred_ratio={_float(row, 'multi_pred_ratio'):.6g}, "
            f"critical_path_len={row['critical_path_len']}, "
            f"strict_ge8={_float(row, 'strict_chain_ge8_coverage'):.6g}, "
            f"bw2_ge8={_float(row, 'bw2_ge8_row_coverage'):.6g}, "
            f"bw4_ge8={_float(row, 'bw4_ge8_row_coverage'):.6g}, "
            f"bw4_ge16={_float(row, 'bw4_ge16_row_coverage'):.6g}, "
            f"bw4_critical={_float(row, 'bw4_critical_path_coverage'):.6g}, "
            f"bw8_ge8={_float(row, 'bw8_ge8_row_coverage'):.6g}, "
            f"bw8_critical={_float(row, 'bw8_critical_path_coverage'):.6g}"
        )

    lines.extend([
        "",
        decision,
        "",
        "Interpretation:",
        "- If strict coverage is low and bw2/bw4 coverage is also low, the scan-chain direction lacks useful structure.",
        "- If only bw8 is strong, the region width is likely too expensive for the original scan-chain premise.",
        "- High frontier-safe coverage alone is not enough; this experiment asks whether bounded-width relaxed regions are large enough.",
    ])

    if failed_rows:
        lines.extend(["", "Failures:"])
        for row in failed_rows:
            lines.append(f"- {row.get('matrix')}: {row.get('error')}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def group_decision(rows: list[dict], group: str) -> str:
    ok_rows = _ok_rows(rows)
    if group == "SPD":
        bw4_ge8 = sum(
            1 for row in ok_rows
            if _float(row, "bw4_ge8_row_coverage") >= 0.10
        )
        bw4_critical = sum(
            1 for row in ok_rows
            if _float(row, "bw4_critical_path_coverage") >= 0.20
        )
        if bw4_ge8 >= 3 or bw4_critical >= 3:
            return "Group decision: SPD has partial relaxed-region support, but needs numerical validation."
        return "Group decision: SPD does not support scan-region direction."
    if group == "CFD":
        support = sum(
            1 for row in ok_rows
            if (
                _float(row, "bw4_ge8_row_coverage") >= 0.10
                or _float(row, "bw4_critical_path_coverage") >= 0.20
            )
        )
        if support >= 3:
            return "Group decision: CFD supports a narrow GMRES/ILU scan-region follow-up."
        return "Group decision: CFD does not provide enough bounded-width scan-region support."
    return final_decision(rows)


def write_plots(fig_dir: Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed; skip plots")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    ok_rows = _ok_rows(rows)
    labels = [row["matrix"] for row in ok_rows]
    groups = [row["group"] for row in ok_rows]

    def save_width_bars(field_template: str, ylabel: str, title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(max(9.0, 0.6 * len(labels)), 5.0))
        x = np.arange(len(labels))
        width_bar = 0.18
        for offset, width in enumerate(WIDTHS):
            values = [
                _float(row, field_template.format(width=width))
                for row in ok_rows
            ]
            ax.bar(
                x + (offset - 1.5) * width_bar,
                values,
                width=width_bar,
                label=f"w={width}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{label}\n{group}" for label, group in zip(labels, groups, strict=False)],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=180)
        plt.close(fig)

    save_width_bars(
        "bw{width}_ge8_row_coverage",
        "row coverage ratio",
        "Bounded-width >=8 region coverage by matrix",
        "bw_ge8_coverage_by_matrix.png",
    )
    save_width_bars(
        "bw{width}_ge16_row_coverage",
        "row coverage ratio",
        "Bounded-width >=16 region coverage by matrix",
        "bw_ge16_coverage_by_matrix.png",
    )
    save_width_bars(
        "bw{width}_critical_path_coverage",
        "critical path coverage",
        "Bounded-width critical path coverage by matrix",
        "bw_critical_path_coverage_by_matrix.png",
    )
    save_width_bars(
        "bw{width}_max_region_len",
        "max region length",
        "Bounded-width max region length by matrix",
        "bw_max_region_len_by_matrix.png",
    )

    for metric, filename, ylabel in [
        ("ge8_row_coverage", "group_mean_ge8_coverage_by_width.png", "mean >=8 coverage"),
        (
            "critical_path_coverage",
            "group_mean_critical_path_coverage_by_width.png",
            "mean critical path coverage",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        x = np.arange(len(WIDTHS))
        bar_width = 0.35
        for group_index, group in enumerate(("SPD", "CFD")):
            values = []
            for width in WIDTHS:
                group_values = [
                    _float(row, f"bw{width}_{metric}")
                    for row in ok_rows
                    if row["group"] == group
                ]
                values.append(_mean(group_values))
            ax.bar(
                x + (group_index - 0.5) * bar_width,
                values,
                width=bar_width,
                label=group,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"w={width}" for width in WIDTHS])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=180)
        plt.close(fig)


def run_experiment(args) -> list[dict]:
    factors = load_formal_factors(
        base.resolve_repo_relative_path(args.spd_manifest),
        base.resolve_repo_relative_path(args.cfd_manifest),
    )
    rows = []
    for factor in factors:
        print(f"[frontier-relaxed] {factor.formal_group}/{factor.matrix_name}", flush=True)
        try:
            rows.append(analyze_factor(factor))
        except Exception as exc:
            print(f"[WARN] {factor.matrix_name} failed: {exc}", flush=True)
            rows.append(failed_row(factor, exc))

    output_path = base.resolve_repo_relative_path(args.out)
    summary_path = base.resolve_repo_relative_path(args.summary_out)
    spd_summary_path = base.resolve_repo_relative_path(args.spd_summary_out)
    cfd_summary_path = base.resolve_repo_relative_path(args.cfd_summary_out)
    fig_dir = base.resolve_repo_relative_path(args.fig_dir)

    write_csv(rows, output_path)
    write_summary(summary_path, rows, "Formal 14 frontier relaxed region summary", group="ALL")
    write_summary(
        spd_summary_path,
        [row for row in rows if row.get("group") == "SPD"],
        "Formal 14 SPD frontier relaxed region summary",
        group="SPD",
    )
    write_summary(
        cfd_summary_path,
        [row for row in rows if row.get("group") == "CFD"],
        "Formal 14 CFD frontier relaxed region summary",
        group="CFD",
    )
    if not args.no_plots:
        write_plots(fig_dir, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run formal 14 frontier/relaxed bounded-width region experiment."
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
        help=f"Total summary path (default: {DEFAULT_SUMMARY})",
    )
    parser.add_argument(
        "--spd-summary-out",
        default=str(DEFAULT_SPD_SUMMARY),
        help=f"SPD summary path (default: {DEFAULT_SPD_SUMMARY})",
    )
    parser.add_argument(
        "--cfd-summary-out",
        default=str(DEFAULT_CFD_SUMMARY),
        help=f"CFD summary path (default: {DEFAULT_CFD_SUMMARY})",
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
    print(f"SPD summary: {base.resolve_repo_relative_path(args.spd_summary_out)}")
    print(f"CFD summary: {base.resolve_repo_relative_path(args.cfd_summary_out)}")
    if not args.no_plots:
        print(f"Figures: {base.resolve_repo_relative_path(args.fig_dir)}")


if __name__ == "__main__":
    main()

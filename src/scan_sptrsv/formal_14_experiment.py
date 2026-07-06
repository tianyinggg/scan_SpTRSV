#!/usr/bin/env python3
"""Formal 14-factor long-chain drop experiment and verdict summaries."""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scan_sptrsv import analyze_scan_chains as base
from scan_sptrsv import drop_experiments as drop


FORMAL_SPD_NAMES = (
    "thermal1",
    "apache1",
    "bcsstk10",
    "bcsstk13",
    "bcsstk15",
    "bcsstk16",
    "bcsstk17",
    "bcsstk18",
)
FORMAL_CFD_NAMES = (
    "cfd1",
    "cfd2",
    "ex11",
    "ex19",
    "ex15",
    "raefsky3",
)

DEFAULT_SPD_MANIFEST = (
    base.RESULTS_DIR
    / "factor_preparation"
    / "suitesparse_spd_structural_thermal_manifest.csv"
)
DEFAULT_CFD_MANIFEST = (
    base.RESULTS_DIR
    / "factor_preparation"
    / "suitesparse_cfd_nonsym_pde_manifest.csv"
)
DEFAULT_FORMAL_OUT = (
    base.RESULTS_DIR
    / "drop_experiments"
    / "formal_14_long_chain_batch.csv"
)
DEFAULT_FORMAL_SUMMARY = (
    base.RESULTS_DIR
    / "drop_experiments"
    / "formal_14_long_chain_batch_summary.txt"
)
DEFAULT_SPD_SUMMARY = (
    base.RESULTS_DIR
    / "drop_experiments"
    / "formal_14_spd_summary.txt"
)
DEFAULT_CFD_SUMMARY = (
    base.RESULTS_DIR
    / "drop_experiments"
    / "formal_14_cfd_summary.txt"
)
DEFAULT_FORMAL_FIG_DIR = (
    base.REPO_ROOT
    / "figures"
    / "drop_experiments"
    / "formal_14_long_chain_batch"
)

FORMAL_EXTRA_FIELDS = [
    "formal_group",
    "long_chain_count_ge_8",
    "long_chain_count_ge_16",
    "long_chain_count_ge_32",
    "chain_gain_per_1pct_drop",
]
FORMAL_CSV_FIELDS = FORMAL_EXTRA_FIELDS + drop.DROP_CSV_FIELDS


@dataclass(frozen=True)
class FormalFactor:
    matrix_name: str
    formal_group: str
    factor_l_path: Path


def _float(row: dict, field: str) -> float:
    value = row.get(field, "")
    if value == "":
        return 0.0
    return float(value)


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _ok_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("status") == "ok"]


def _nonbaseline_rows(rows: list[dict]) -> list[dict]:
    return [
        row for row in _ok_rows(rows)
        if row.get("drop_method") != "no-drop"
    ]


def _long_metric_key(row: dict) -> tuple[float, float, float, float, float]:
    return (
        _float(row, "effective_chain_coverage_ge_16_gain"),
        _float(row, "effective_chain_coverage_ge_8_gain"),
        _float(row, "effective_chain_coverage_ge_32_gain"),
        _float(row, "effective_long_chain_score_gain_per_drop_percent"),
        _float(row, "effective_chain_coverage_ge_8_gain_per_drop_percent"),
    )


def _best_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return max(rows, key=_long_metric_key)


def _matrix_names(rows: list[dict]) -> list[str]:
    return sorted({row["matrix_name"] for row in _ok_rows(rows)})


def _method_rows(rows: list[dict], method: str) -> list[dict]:
    return [row for row in _ok_rows(rows) if row.get("drop_method") == method]


def _best_by_matrix_and_method(rows: list[dict], method: str) -> dict[str, dict]:
    best = {}
    for matrix in _matrix_names(rows):
        matrix_rows = [
            row for row in _ok_rows(rows)
            if row["matrix_name"] == matrix and row["drop_method"] == method
        ]
        selected = _best_row(matrix_rows)
        if selected is not None:
            best[matrix] = selected
    return best


def _best_any_by_matrix(rows: list[dict]) -> dict[str, dict]:
    best = {}
    for matrix in _matrix_names(rows):
        selected = _best_row([
            row for row in _nonbaseline_rows(rows)
            if row["matrix_name"] == matrix
        ])
        if selected is not None:
            best[matrix] = selected
    return best


def _verdict(row: dict | None) -> str:
    if row is None:
        return "failed"
    ge8_gain = _float(row, "effective_chain_coverage_ge_8_gain")
    ge16_gain = _float(row, "effective_chain_coverage_ge_16_gain")
    ge32_gain = _float(row, "effective_chain_coverage_ge_32_gain")
    score = _float(row, "effective_long_chain_score_gain_per_drop_percent")
    if ge16_gain >= 0.005 or ge8_gain >= 0.01 or ge32_gain >= 0.002:
        return "effective"
    if ge16_gain < -1e-9 or ge8_gain < -1e-9 or ge32_gain < -1e-9 or score < -1e-9:
        return "regressed"
    return "weak-or-ineffective"


def _formalize_row(row: dict, formal_group: str) -> dict:
    formal = dict(row)
    formal["formal_group"] = formal_group
    formal["long_chain_count_ge_8"] = row.get("effective_chain_count_ge_8_after", "")
    formal["long_chain_count_ge_16"] = row.get("effective_chain_count_ge_16_after", "")
    formal["long_chain_count_ge_32"] = row.get("effective_chain_count_ge_32_after", "")
    formal["chain_gain_per_1pct_drop"] = row.get(
        "effective_long_chain_score_gain_per_drop_percent",
        "",
    )
    return formal


def _load_manifest_factors(
    manifest_path: Path,
    formal_group: str,
    allowed_names: tuple[str, ...],
) -> list[FormalFactor]:
    by_name = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["matrix_name"]
            if name not in allowed_names:
                continue
            factor_path = base.resolve_repo_relative_path(row["factor_l_path"])
            by_name[name] = FormalFactor(
                matrix_name=name,
                formal_group=formal_group,
                factor_l_path=factor_path,
            )

    missing = [name for name in allowed_names if name not in by_name]
    if missing:
        raise ValueError(
            f"{manifest_path} is missing formal factors: {', '.join(missing)}"
        )
    missing_files = [
        str(by_name[name].factor_l_path)
        for name in allowed_names
        if not by_name[name].factor_l_path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Formal factor paths not found: " + ", ".join(missing_files)
        )
    return [by_name[name] for name in allowed_names]


def load_formal_factors(spd_manifest: Path, cfd_manifest: Path) -> list[FormalFactor]:
    spd = _load_manifest_factors(spd_manifest, "SPD", FORMAL_SPD_NAMES)
    cfd = _load_manifest_factors(cfd_manifest, "CFD", FORMAL_CFD_NAMES)
    return spd + cfd


def _write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FORMAL_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FORMAL_CSV_FIELDS})


def _run_one_factor(
    factor: FormalFactor,
    configs: list[drop.DropConfig],
) -> list[dict]:
    graph = drop.read_valued_strict_lower_graph_mtx(factor.factor_l_path)
    context = drop.build_drop_context(graph)
    rows = []
    for config in configs:
        row = drop.drop_experiment_row(
            matrix_name=factor.matrix_name,
            source="factor-l",
            input_path=factor.factor_l_path,
            before_graph=graph,
            context=context,
            config=config,
        )
        rows.append(_formalize_row(row, factor.formal_group))
    return rows


def _failed_row(factor: FormalFactor, exc: Exception) -> dict:
    row = {
        "formal_group": factor.formal_group,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "matrix_name": factor.matrix_name,
        "source": "factor-l",
        "input_path": str(factor.factor_l_path),
    }
    return row


def run_formal_experiment(args) -> list[dict]:
    spd_manifest = base.resolve_repo_relative_path(args.spd_manifest)
    cfd_manifest = base.resolve_repo_relative_path(args.cfd_manifest)
    factors = load_formal_factors(spd_manifest, cfd_manifest)

    methods = drop.parse_method_list(args.methods)
    drop_ratios = drop.parse_float_list(args.drop_ratios)
    risk_thresholds = drop.parse_float_list(args.risk_thresholds)
    max_drop_per_row = drop.parse_int_list(args.max_drop_per_row)
    configs = list(drop.iter_drop_configs(
        methods=methods,
        drop_ratios=drop_ratios,
        risk_thresholds=risk_thresholds,
        max_drop_per_row_values=max_drop_per_row,
        keep_row_max_predecessor=not args.allow_drop_row_max_predecessor,
        forbid_empty_rows=not args.allow_empty_rows,
    ))

    all_rows = []
    for factor in factors:
        print(f"[formal-14] {factor.formal_group}/{factor.matrix_name}", flush=True)
        try:
            all_rows.extend(_run_one_factor(factor, configs))
        except Exception as exc:
            print(f"[WARN] {factor.matrix_name} failed: {exc}", flush=True)
            all_rows.append(_failed_row(factor, exc))

    output_path = base.resolve_repo_relative_path(args.out)
    summary_path = base.resolve_repo_relative_path(args.summary_out)
    spd_summary_path = base.resolve_repo_relative_path(args.spd_summary_out)
    cfd_summary_path = base.resolve_repo_relative_path(args.cfd_summary_out)
    fig_dir = base.resolve_repo_relative_path(args.fig_dir)

    _write_csv(all_rows, output_path)
    write_formal_summary(summary_path, all_rows, title="Formal 14-factor verdict")
    write_group_summary(
        spd_summary_path,
        [row for row in all_rows if row.get("formal_group") == "SPD"],
        group="SPD",
    )
    write_group_summary(
        cfd_summary_path,
        [row for row in all_rows if row.get("formal_group") == "CFD"],
        group="CFD",
    )
    if not args.no_plots:
        write_formal_plots(fig_dir, all_rows)

    return all_rows


def _method_stats(rows: list[dict]) -> list[str]:
    lines = []
    methods = sorted({row["drop_method"] for row in _ok_rows(rows)})
    for method in methods:
        method_rows = _method_rows(rows, method)
        if not method_rows:
            continue
        ge8 = [_float(row, "effective_chain_coverage_ge_8_gain") for row in method_rows]
        ge16 = [_float(row, "effective_chain_coverage_ge_16_gain") for row in method_rows]
        ge32 = [_float(row, "effective_chain_coverage_ge_32_gain") for row in method_rows]
        gain = [_float(row, "chain_gain_per_1pct_drop") for row in method_rows]
        lines.append(
            f"- {method}: mean_ge8_gain={_mean(ge8):.6g}, median_ge8_gain={_median(ge8):.6g}, "
            f"mean_ge16_gain={_mean(ge16):.6g}, median_ge16_gain={_median(ge16):.6g}, "
            f"mean_ge32_gain={_mean(ge32):.6g}, median_ge32_gain={_median(ge32):.6g}, "
            f"mean_chain_gain_per_1pct_drop={_mean(gain):.6g}"
        )
    return lines


def _long_vs_magnitude(rows: list[dict]) -> tuple[int, int, int, list[float], list[float]]:
    best_long = _best_by_matrix_and_method(rows, "long-chain-aware-drop")
    best_mag = _best_by_matrix_and_method(rows, "magnitude-drop")
    wins = losses = ties = 0
    ge8_deltas = []
    ge16_deltas = []
    for matrix, long_row in best_long.items():
        mag_row = best_mag.get(matrix)
        if mag_row is None:
            continue
        long_key = _long_metric_key(long_row)
        mag_key = _long_metric_key(mag_row)
        if long_key > mag_key:
            wins += 1
        elif long_key < mag_key:
            losses += 1
        else:
            ties += 1
        ge8_deltas.append(
            _float(long_row, "effective_chain_coverage_ge_8_gain")
            - _float(mag_row, "effective_chain_coverage_ge_8_gain")
        )
        ge16_deltas.append(
            _float(long_row, "effective_chain_coverage_ge_16_gain")
            - _float(mag_row, "effective_chain_coverage_ge_16_gain")
        )
    return wins, losses, ties, ge8_deltas, ge16_deltas


def _verdict_counts(rows: list[dict], method: str = "long-chain-aware-drop") -> dict[str, int]:
    best = _best_by_matrix_and_method(rows, method)
    counts = {"effective": 0, "weak-or-ineffective": 0, "regressed": 0, "failed": 0}
    for matrix in _matrix_names(rows):
        counts[_verdict(best.get(matrix))] += 1
    failed_matrices = {
        row["matrix_name"] for row in rows if row.get("status") == "failed"
    }
    counts["failed"] += len(failed_matrices)
    return counts


def _decision(rows: list[dict]) -> str:
    all_counts = _verdict_counts(rows)
    spd_counts = _verdict_counts([
        row for row in rows if row.get("formal_group") == "SPD"
    ])
    cfd_counts = _verdict_counts([
        row for row in rows if row.get("formal_group") == "CFD"
    ])
    wins, losses, _, ge8_deltas, ge16_deltas = _long_vs_magnitude(rows)
    long_beats_mag = (
        wins > losses
        and (_median(ge8_deltas) > 0 or _median(ge16_deltas) > 0)
    )
    if (
        all_counts["effective"] >= 4
        and cfd_counts["effective"] >= 2
        and long_beats_mag
    ):
        return "A. 支持继续：先做 CPU 数值收敛验证，不直接写 GPU kernel。"
    if (
        spd_counts["effective"] <= 1
        and cfd_counts["effective"] >= 2
        and cfd_counts["effective"] > spd_counts["effective"]
    ):
        return "B. 只支持窄场景：收窄到非对称 PDE / GMRES-ILU 预条件器场景。"
    return "C. 不支持继续：不支持进入 scan-chain executor，应停止 scan 链作为主线。"


def write_formal_summary(summary_path: Path, rows: list[dict], title: str) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    ok_rows = _ok_rows(rows)
    matrices = _matrix_names(rows)
    best_long = _best_by_matrix_and_method(rows, "long-chain-aware-drop")
    best_any = _best_any_by_matrix(rows)
    counts = _verdict_counts(rows, "long-chain-aware-drop")
    best_any_counts = {
        "effective": 0,
        "weak-or-ineffective": 0,
        "regressed": 0,
        "failed": 0,
    }
    for matrix in matrices:
        best_any_counts[_verdict(best_any.get(matrix))] += 1

    wins, losses, ties, ge8_deltas, ge16_deltas = _long_vs_magnitude(rows)
    ge8_effective = sum(
        1 for row in best_long.values()
        if _float(row, "effective_chain_coverage_ge_8_gain") >= 0.01
    )
    ge16_effective = sum(
        1 for row in best_long.values()
        if _float(row, "effective_chain_coverage_ge_16_gain") >= 0.005
    )
    best_ge8_any = sum(
        1 for row in best_any.values()
        if _float(row, "effective_chain_coverage_ge_8_gain") >= 0.01
    )
    best_ge16_any = sum(
        1 for row in best_any.values()
        if _float(row, "effective_chain_coverage_ge_16_gain") >= 0.005
    )

    lines = [
        title,
        "",
        "Scope:",
        "- Only formal manifest factors are included.",
        f"- matrices={len(matrices)}, ok_rows={len(ok_rows)}, failed_rows={len(rows) - len(ok_rows)}",
        "- Formal SPD set: " + ", ".join(FORMAL_SPD_NAMES),
        "- Formal CFD set: " + ", ".join(FORMAL_CFD_NAMES),
        "",
        "Decision metric:",
        "- Primary metrics are effective_chain_coverage_ge_8/16/32_gain and long-chain row/count fields.",
        "- strict_chain_row_coverage_gain is auxiliary and is not used as direction evidence.",
        "- chain_gain_per_1pct_drop is effective_long_chain_score_gain_per_drop_percent.",
        "",
        "Long-chain-aware-drop verdict counts:",
        f"- effective: {counts['effective']}",
        f"- weak-or-ineffective: {counts['weak-or-ineffective']}",
        f"- regressed: {counts['regressed']}",
        f"- failed: {counts['failed']}",
        "",
        "Best-any-method verdict counts:",
        f"- effective: {best_any_counts['effective']}",
        f"- weak-or-ineffective: {best_any_counts['weak-or-ineffective']}",
        f"- regressed: {best_any_counts['regressed']}",
        "",
        "Required direct answers for all 14 factors:",
        f"- long-chain-aware-drop effective matrices: {counts['effective']} / {len(matrices)}",
        f"- weak-or-ineffective matrices: {counts['weak-or-ineffective']} / {len(matrices)}",
        f"- regressed matrices: {counts['regressed']} / {len(matrices)}",
        (
            "- long-chain-aware-drop stable vs magnitude-drop: "
            f"wins={wins}, losses={losses}, ties={ties}, "
            f"median_ge8_delta={_median(ge8_deltas):.6g}, "
            f"median_ge16_delta={_median(ge16_deltas):.6g}"
        ),
        (
            "- matrices with obvious >=8 gain under long-chain-aware-drop: "
            f"{ge8_effective}; best-any-method: {best_ge8_any}"
        ),
        (
            "- matrices with obvious >=16 gain under long-chain-aware-drop: "
            f"{ge16_effective}; best-any-method: {best_ge16_any}"
        ),
        f"- current next-stage decision: {_decision(rows)}",
        "",
        "Method aggregate statistics:",
        *_method_stats(rows),
        "",
        "Per-matrix best long-chain-aware-drop rows:",
    ]

    for matrix in matrices:
        row = best_long.get(matrix)
        if row is None:
            lines.append(f"- {matrix}: failed/no row")
            continue
        lines.append(
            f"- {matrix} [{row['formal_group']}]: {_verdict(row)}; "
            f"ratio={row['drop_ratio_target']}, risk={row['risk_threshold']}, "
            f"max_drop_per_row={row['max_drop_per_row']}, "
            f"drop_ratio_actual={_float(row, 'drop_ratio_actual'):.6g}, "
            f"dropped_nnz={row['dropped_nnz']}, "
            f"ge8_gain={_float(row, 'effective_chain_coverage_ge_8_gain'):.6g}, "
            f"ge16_gain={_float(row, 'effective_chain_coverage_ge_16_gain'):.6g}, "
            f"ge32_gain={_float(row, 'effective_chain_coverage_ge_32_gain'):.6g}, "
            f"rows_ge8_after={row['effective_chain_rows_ge_8_after']}, "
            f"rows_ge16_after={row['effective_chain_rows_ge_16_after']}, "
            f"rows_ge32_after={row['effective_chain_rows_ge_32_after']}, "
            f"count_ge8_after={row['long_chain_count_ge_8']}, "
            f"count_ge16_after={row['long_chain_count_ge_16']}, "
            f"count_ge32_after={row['long_chain_count_ge_32']}, "
            f"chain_gain_per_1pct_drop={_float(row, 'chain_gain_per_1pct_drop'):.6g}"
        )

    lines.extend([
        "",
        "Per-matrix best method rows:",
    ])
    for matrix in matrices:
        row = best_any.get(matrix)
        if row is None:
            lines.append(f"- {matrix}: failed/no row")
            continue
        lines.append(
            f"- {matrix} [{row['formal_group']}]: best={row['drop_method']}, "
            f"verdict={_verdict(row)}, "
            f"ratio={row['drop_ratio_target']}, risk={row['risk_threshold']}, "
            f"ge8_gain={_float(row, 'effective_chain_coverage_ge_8_gain'):.6g}, "
            f"ge16_gain={_float(row, 'effective_chain_coverage_ge_16_gain'):.6g}, "
            f"ge32_gain={_float(row, 'effective_chain_coverage_ge_32_gain'):.6g}"
        )

    lines.extend([
        "",
        "Decision rule:",
        "- effective if ge16 coverage gain >= 0.005, or ge8 coverage gain >= 0.01, or ge32 coverage gain >= 0.002.",
        "- regressed if the best row has negative ge8/ge16/ge32 gain or negative weighted long-chain gain per 1% drop.",
        "- otherwise weak-or-ineffective.",
    ])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_group_summary(summary_path: Path, rows: list[dict], group: str) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    counts = _verdict_counts(rows, "long-chain-aware-drop")
    best_long = _best_by_matrix_and_method(rows, "long-chain-aware-drop")
    matrices = _matrix_names(rows)
    wins, losses, ties, ge8_deltas, ge16_deltas = _long_vs_magnitude(rows)
    best_values = list(best_long.values())
    ge8_mean = _mean([
        _float(row, "effective_chain_coverage_ge_8_gain")
        for row in best_values
    ])
    ge16_mean = _mean([
        _float(row, "effective_chain_coverage_ge_16_gain")
        for row in best_values
    ])

    lines = [
        f"Formal 14 {group} group summary",
        "",
        f"matrices: {len(matrices)}",
        f"long-chain-aware effective: {counts['effective']}",
        f"weak-or-ineffective: {counts['weak-or-ineffective']}",
        f"regressed: {counts['regressed']}",
        (
            "long-chain-aware vs magnitude-drop: "
            f"wins={wins}, losses={losses}, ties={ties}, "
            f"median_ge8_delta={_median(ge8_deltas):.6g}, "
            f"median_ge16_delta={_median(ge16_deltas):.6g}"
        ),
        f"mean best long-chain-aware ge8 gain: {ge8_mean:.6g}",
        f"mean best long-chain-aware ge16 gain: {ge16_mean:.6g}",
        "",
        "Method aggregate statistics:",
        *_method_stats(rows),
        "",
        "Per-matrix long-chain-aware best:",
    ]
    for matrix in matrices:
        row = best_long.get(matrix)
        if row is None:
            lines.append(f"- {matrix}: failed/no row")
            continue
        lines.append(
            f"- {matrix}: {_verdict(row)}, "
            f"ge8_gain={_float(row, 'effective_chain_coverage_ge_8_gain'):.6g}, "
            f"ge16_gain={_float(row, 'effective_chain_coverage_ge_16_gain'):.6g}, "
            f"ge32_gain={_float(row, 'effective_chain_coverage_ge_32_gain'):.6g}, "
            f"best_ratio={row['drop_ratio_target']}, "
            f"best_risk={row['risk_threshold']}, "
            f"chain_gain_per_1pct_drop={_float(row, 'chain_gain_per_1pct_drop'):.6g}"
        )

    lines.extend(["", "Direct group answers:"])
    if group == "SPD":
        bcsstk_rows = [
            row for row in rows
            if row.get("matrix_name", "").startswith("bcsstk")
        ]
        bcsstk_counts = _verdict_counts(bcsstk_rows, "long-chain-aware-drop")
        thermal_apache = [
            row for row in best_values
            if row["matrix_name"] in {"thermal1", "apache1"}
        ]
        bcsstk_best = [
            row for row in best_values
            if row["matrix_name"].startswith("bcsstk")
        ]
        ta_ge8 = _mean([
            _float(row, "effective_chain_coverage_ge_8_gain")
            for row in thermal_apache
        ])
        bcsstk_ge8 = _mean([
            _float(row, "effective_chain_coverage_ge_8_gain")
            for row in bcsstk_best
        ])
        if counts["effective"] >= 2:
            lines.append("- SPD / structural / thermal has some scan-chain signal, but judge the per-matrix details before executor work.")
        else:
            lines.append("- SPD / structural / thermal is not broadly suitable for manufacturing useful long scan chains under the tested deletion rules.")
        lines.append(
            f"- bcsstk structural stiffness matrices: effective={bcsstk_counts['effective']}, "
            f"weak={bcsstk_counts['weak-or-ineffective']}, regressed={bcsstk_counts['regressed']}."
        )
        if bcsstk_counts["effective"] == 0:
            lines.append("- bcsstk class is broadly ineffective in this batch.")
        else:
            lines.append("- bcsstk class has isolated support, not a universal positive result.")
        lines.append(
            f"- thermal1/apache1 mean ge8 gain={ta_ge8:.6g}; bcsstk mean ge8 gain={bcsstk_ge8:.6g}."
        )
        if ta_ge8 > bcsstk_ge8:
            lines.append("- thermal1/apache1 differ from bcsstk by showing stronger >=8 long-chain gain.")
        else:
            lines.append("- thermal1/apache1 do not show a clearly stronger >=8 long-chain signal than bcsstk.")
    else:
        cfd_effective = counts["effective"]
        if cfd_effective >= 2:
            lines.append("- CFD / nonsymmetric PDE is more promising than a generic SPD interpretation if SPD does not match this effective count.")
        else:
            lines.append("- CFD / nonsymmetric PDE does not show enough stable long-chain improvement by itself.")
        lines.append(
            "- COLAMD ILU L factors in this group are the tested objects; this result cannot be interpreted as a NATURAL-vs-COLAMD comparison."
        )
        if cfd_effective >= 2 and wins >= losses:
            lines.append("- If the full summary also shows SPD weak, the research scope should narrow to GMRES/ILU nonsymmetric PDE cases.")
        else:
            lines.append("- Current CFD evidence is not strong enough to justify narrowing the whole project around GMRES/ILU yet.")

    lines.extend([
        "",
        "Decision rule:",
        "- effective if ge16 coverage gain >= 0.005, or ge8 coverage gain >= 0.01, or ge32 coverage gain >= 0.002.",
        "- strict total coverage is auxiliary only.",
    ])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_formal_plots(fig_dir: Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed; skip formal plots")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    ok_rows = _ok_rows(rows)
    best_any = _best_any_by_matrix(rows)
    labels = [matrix for matrix in _matrix_names(rows) if matrix in best_any]

    def save_bar(values, methods, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(max(8.5, 0.55 * len(labels)), 4.8))
        x = np.arange(len(labels))
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{label}\n{method}" for label, method in zip(labels, methods, strict=False)],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=180)
        plt.close(fig)

    if labels:
        save_bar(
            [
                _float(best_any[matrix], "effective_chain_coverage_ge_8_gain")
                for matrix in labels
            ],
            [best_any[matrix]["drop_method"] for matrix in labels],
            "Best >=8 effective chain coverage gain per matrix",
            "coverage gain",
            "best_ge8_coverage_gain_by_matrix.png",
        )
        save_bar(
            [
                _float(best_any[matrix], "effective_chain_coverage_ge_16_gain")
                for matrix in labels
            ],
            [best_any[matrix]["drop_method"] for matrix in labels],
            "Best >=16 effective chain coverage gain per matrix",
            "coverage gain",
            "best_ge16_coverage_gain_by_matrix.png",
        )
        save_bar(
            [
                _float(best_any[matrix], "chain_gain_per_1pct_drop")
                for matrix in labels
            ],
            [best_any[matrix]["drop_method"] for matrix in labels],
            "Best weighted long-chain gain per dropped 1%",
            "weighted gain per dropped 1%",
            "best_chain_gain_per_1pct_drop_by_matrix.png",
        )
        save_bar(
            [
                _float(best_any[matrix], "effective_chain_coverage_ge_8_gain")
                for matrix in labels
            ],
            [best_any[matrix]["drop_method"] for matrix in labels],
            "Best method by matrix",
            "best >=8 coverage gain",
            "best_method_by_matrix.png",
        )

    for field, filename, title in [
        (
            "effective_chain_coverage_ge_8_gain",
            "method_group_mean_median_ge8_gain.png",
            "Method mean/median >=8 gain by group",
        ),
        (
            "effective_chain_coverage_ge_16_gain",
            "method_group_mean_median_ge16_gain.png",
            "Method mean/median >=16 gain by group",
        ),
    ]:
        methods = [method for method in drop.DROP_METHODS if method != "no-drop"]
        groups = ["SPD", "CFD"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        for ax, group in zip(axes, groups, strict=False):
            means = []
            medians = []
            for method in methods:
                values = [
                    _float(row, field)
                    for row in ok_rows
                    if row["formal_group"] == group and row["drop_method"] == method
                ]
                means.append(_mean(values))
                medians.append(_median(values))
            x = np.arange(len(methods))
            width = 0.35
            ax.bar(x - width / 2, means, width=width, label="mean")
            ax.bar(x + width / 2, medians, width=width, label="median")
            ax.set_title(group)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=25, ha="right")
            ax.grid(True, axis="y", alpha=0.3)
        axes[0].set_ylabel("coverage gain")
        axes[1].legend()
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=180)
        plt.close(fig)

    best_long = _best_by_matrix_and_method(rows, "long-chain-aware-drop")
    best_mag = _best_by_matrix_and_method(rows, "magnitude-drop")
    delta_labels = [
        matrix for matrix in _matrix_names(rows)
        if matrix in best_long and matrix in best_mag
    ]
    if delta_labels:
        fig, ax = plt.subplots(figsize=(max(8.5, 0.55 * len(delta_labels)), 4.8))
        x = np.arange(len(delta_labels))
        width = 0.35
        ge8_delta = [
            _float(best_long[matrix], "effective_chain_coverage_ge_8_gain")
            - _float(best_mag[matrix], "effective_chain_coverage_ge_8_gain")
            for matrix in delta_labels
        ]
        ge16_delta = [
            _float(best_long[matrix], "effective_chain_coverage_ge_16_gain")
            - _float(best_mag[matrix], "effective_chain_coverage_ge_16_gain")
            for matrix in delta_labels
        ]
        ax.bar(x - width / 2, ge8_delta, width=width, label=">=8 delta")
        ax.bar(x + width / 2, ge16_delta, width=width, label=">=16 delta")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(delta_labels, rotation=25, ha="right")
        ax.set_ylabel("long-chain-aware minus magnitude-drop")
        ax.set_title("Long-chain-aware-drop relative gain distribution")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "long_chain_aware_vs_magnitude_distribution.png", dpi=180)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the formal 14-factor long-chain-aware drop verdict experiment."
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
        "--methods",
        default=",".join(drop.DROP_METHODS),
        help=f"Comma-separated methods (default: {','.join(drop.DROP_METHODS)})",
    )
    parser.add_argument(
        "--drop-ratios",
        default="0.01,0.02,0.05,0.10",
        help="Comma-separated target drop ratios (default: 0.01,0.02,0.05,0.10)",
    )
    parser.add_argument(
        "--risk-thresholds",
        default="1e-4,1e-3,1e-2",
        help="Comma-separated risk thresholds (default: 1e-4,1e-3,1e-2)",
    )
    parser.add_argument(
        "--max-drop-per-row",
        default="1,-1",
        help="Comma-separated per-row drop limits; -1 means unlimited (default: 1,-1)",
    )
    parser.add_argument(
        "--allow-drop-row-max-predecessor",
        action="store_true",
        help="Allow deleting the largest-magnitude predecessor in each row.",
    )
    parser.add_argument(
        "--allow-empty-rows",
        action="store_true",
        help="Allow deleting all strict-lower predecessors of a row.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_FORMAL_OUT),
        help=f"Formal CSV output (default: {DEFAULT_FORMAL_OUT})",
    )
    parser.add_argument(
        "--summary-out",
        default=str(DEFAULT_FORMAL_SUMMARY),
        help=f"Formal total summary output (default: {DEFAULT_FORMAL_SUMMARY})",
    )
    parser.add_argument(
        "--spd-summary-out",
        default=str(DEFAULT_SPD_SUMMARY),
        help=f"SPD group summary output (default: {DEFAULT_SPD_SUMMARY})",
    )
    parser.add_argument(
        "--cfd-summary-out",
        default=str(DEFAULT_CFD_SUMMARY),
        help=f"CFD group summary output (default: {DEFAULT_CFD_SUMMARY})",
    )
    parser.add_argument(
        "--fig-dir",
        default=str(DEFAULT_FORMAL_FIG_DIR),
        help=f"Formal figure directory (default: {DEFAULT_FORMAL_FIG_DIR})",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip formal plot generation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_formal_experiment(args)
    print(f"Done. Formal rows: {len(rows)}")
    print(f"CSV: {base.resolve_repo_relative_path(args.out)}")
    print(f"Summary: {base.resolve_repo_relative_path(args.summary_out)}")
    print(f"SPD summary: {base.resolve_repo_relative_path(args.spd_summary_out)}")
    print(f"CFD summary: {base.resolve_repo_relative_path(args.cfd_summary_out)}")
    if not args.no_plots:
        print(f"Figures: {base.resolve_repo_relative_path(args.fig_dir)}")


if __name__ == "__main__":
    main()

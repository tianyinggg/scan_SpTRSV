#!/usr/bin/env python3
import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scan_sptrsv import analyze_scan_chains as base


DROP_METHODS = (
    "no-drop",
    "magnitude-drop",
    "wavefront-oriented",
    "scan-chain-aware",
    "long-chain-aware-drop",
)

DEFAULT_DROP_RATIOS = (0.01, 0.02, 0.05, 0.10)
DEFAULT_RISK_THRESHOLDS = (1e-4, 1e-3, 1e-2)
DEFAULT_MAX_DROP_PER_ROW = (1,)
DROP_RESULTS_DIR = base.RESULTS_DIR / "drop_experiments"
DEFAULT_DROP_OUT = DROP_RESULTS_DIR / "drop_scan_chain_sweep.csv"
DEFAULT_DROP_SUMMARY_OUT = DROP_RESULTS_DIR / "drop_scan_chain_summary.txt"
DEFAULT_DROP_FIG_DIR = base.REPO_ROOT / "figures" / "drop_experiments"
DEFAULT_DROP_DEFINITIONS_OUT = (
    base.RESULTS_DIR / "definitions" / "drop_experiment_fields.csv"
)


DROP_CSV_FIELDS = [
    "status",
    "error",
    "matrix_name",
    "source",
    "input_path",
    "n_rows",
    "drop_method",
    "drop_ratio_target",
    "drop_ratio_actual",
    "risk_threshold",
    "max_drop_per_row",
    "keep_row_max_predecessor",
    "forbid_empty_rows",
    "dropped_nnz",
    "dropped_rows",
    "strict_lower_nnz_before",
    "strict_lower_nnz_after",
    "single_pred_rows_before",
    "single_pred_rows_after",
    "multi_pred_rows_before",
    "multi_pred_rows_after",
    "multi_pred_to_single_pred_rows",
    "strict_chain_count_before",
    "strict_chain_count_after",
    "strict_chain_row_coverage_before",
    "strict_chain_row_coverage_after",
    "strict_chain_row_coverage_gain",
    "strict_chain_row_coverage_gain_ratio",
    "strict_chain_coverage_gain_per_drop_ratio",
    "strict_chain_coverage_gain_per_drop_percent",
    "frontier_safe_chain_row_coverage_before",
    "frontier_safe_chain_row_coverage_after",
    "frontier_safe_chain_row_coverage_gain",
    "frontier_safe_chain_row_coverage_gain_ratio",
    "frontier_safe_boundary_cut_before",
    "frontier_safe_boundary_cut_after",
    "frontier_safe_boundary_cut_reduction",
    "effective_chain_rows_ge_8_before",
    "effective_chain_rows_ge_8_after",
    "effective_chain_rows_ge_8",
    "effective_chain_rows_ge_8_gain",
    "effective_chain_rows_ge_8_gain_per_dropped_edge",
    "effective_chain_rows_ge_16_before",
    "effective_chain_rows_ge_16_after",
    "effective_chain_rows_ge_16",
    "effective_chain_rows_ge_16_gain",
    "effective_chain_rows_ge_16_gain_per_dropped_edge",
    "effective_chain_rows_ge_32_before",
    "effective_chain_rows_ge_32_after",
    "effective_chain_rows_ge_32",
    "effective_chain_rows_ge_32_gain",
    "effective_chain_rows_ge_32_gain_per_dropped_edge",
    "effective_chain_count_ge_8_before",
    "effective_chain_count_ge_8_after",
    "effective_chain_count_ge_8_gain",
    "effective_chain_count_ge_16_before",
    "effective_chain_count_ge_16_after",
    "effective_chain_count_ge_16_gain",
    "effective_chain_count_ge_32_before",
    "effective_chain_count_ge_32_after",
    "effective_chain_count_ge_32_gain",
    "effective_chain_coverage_ge_8_before",
    "effective_chain_coverage_ge_8_after",
    "effective_chain_coverage_ge_8",
    "effective_chain_coverage_ge_8_gain",
    "effective_chain_coverage_ge_8_gain_per_drop_ratio",
    "effective_chain_coverage_ge_8_gain_per_drop_percent",
    "effective_chain_coverage_ge_16_before",
    "effective_chain_coverage_ge_16_after",
    "effective_chain_coverage_ge_16",
    "effective_chain_coverage_ge_16_gain",
    "effective_chain_coverage_ge_16_gain_per_drop_ratio",
    "effective_chain_coverage_ge_16_gain_per_drop_percent",
    "effective_chain_coverage_ge_32_before",
    "effective_chain_coverage_ge_32_after",
    "effective_chain_coverage_ge_32",
    "effective_chain_coverage_ge_32_gain",
    "effective_chain_coverage_ge_32_gain_per_drop_ratio",
    "effective_chain_coverage_ge_32_gain_per_drop_percent",
    "avg_chain_length_before",
    "avg_chain_length_after",
    "avg_effective_chain_length_ge_8_before",
    "avg_effective_chain_length_ge_8_after",
    "avg_effective_chain_length_ge_16_before",
    "avg_effective_chain_length_ge_16_after",
    "avg_effective_chain_length_ge_32_before",
    "avg_effective_chain_length_ge_32_after",
    "effective_long_chain_score_before",
    "effective_long_chain_score_after",
    "effective_long_chain_score_gain",
    "effective_long_chain_score_gain_per_drop_percent",
    "max_chain_length_before",
    "max_chain_length_after",
    "strict_chain_len_ge_8_before",
    "strict_chain_len_ge_8_after",
    "strict_chain_len_ge_16_before",
    "strict_chain_len_ge_16_after",
    "strict_chain_len_ge_32_before",
    "strict_chain_len_ge_32_after",
    "chain_gain_per_dropped_edge",
]

DROP_UPSERT_KEY_FIELDS = [
    "matrix_name",
    "source",
    "input_path",
    "drop_method",
    "drop_ratio_target",
    "risk_threshold",
    "max_drop_per_row",
    "keep_row_max_predecessor",
    "forbid_empty_rows",
]


DROP_FIELD_DEFINITIONS = {
    "status": (
        "状态",
        "单个矩阵和参数组合是否成功完成；失败时仍写入一行。",
        "ok 表示完成，failed 表示该矩阵失败但批处理继续。",
        "status ∈ {ok, failed}",
    ),
    "error": (
        "错误信息",
        "失败矩阵记录 Python 异常文本；成功行为空。",
        "用于批处理后定位无法读取、无法因子化或内存不足的输入。",
        "error = str(exception) if failed else ''",
    ),
    "matrix_name": (
        "矩阵名",
        "默认来自输入 .mtx 文件名；spcg-ilu-l 来自生成的 L 因子名。",
        "区分不同实验对象。",
        "matrix_name = Path(input).stem 或 generated_factor_name",
    ),
    "source": (
        "数据来源",
        "由 --source 指定，可为 factor-l、spcg-ilu-l、matrix-lower。",
        "标记当前统计对象是真实 L 因子、生成 L 因子还是原矩阵下三角。",
        "source ∈ {factor-l, spcg-ilu-l, matrix-lower}",
    ),
    "input_path": (
        "输入路径",
        "实际被分析的 .mtx 路径；spcg-ilu-l 时为生成后的 L 因子路径。",
        "用于复现实验输入。",
        "input_path = analysis_path",
    ),
    "n_rows": (
        "行数",
        "从 Matrix Market 尺寸行读取，要求方阵。",
        "三角系统变量数，也是覆盖率分母。",
        "n_rows = M = N",
    ),
    "drop_method": (
        "删元素方法",
        "当前参数组合使用的删边策略。",
        "比较 no-drop、普通删小值、波前导向和 scan-chain-aware 的核心字段。",
        "drop_method ∈ DROP_METHODS",
    ),
    "drop_ratio_target": (
        "目标删除比例",
        "命令行 --drop-ratios 给定的目标 strict-lower 非零元删除比例。",
        "控制同等删除预算下的比较。",
        "target = requested_drop_nnz / strict_lower_nnz_before",
    ),
    "drop_ratio_actual": (
        "实际删除比例",
        "实际删除的 strict-lower 非零元数除以删除前 strict_lower_nnz。",
        "受 risk threshold、每行删除上限、禁止删空等约束影响，可能小于目标比例。",
        "drop_ratio_actual = dropped_nnz / strict_lower_nnz_before",
    ),
    "risk_threshold": (
        "数值风险阈值",
        "只允许删除 abs(a_ij)/max_abs(row_i) 不超过该阈值的边。",
        "控制删边数值风险。",
        "risk = |a_ij| / max_j |a_ij|",
    ),
    "max_drop_per_row": (
        "每行最大删除数",
        "每一行最多删除多少个 strict-lower 前驱；-1 表示不限。",
        "避免在单行中过度删边。",
        "row_drop_count[i] <= max_drop_per_row",
    ),
    "keep_row_max_predecessor": (
        "保留每行最大幅值前驱",
        "为 1 时不删除该行绝对值最大的 strict-lower 前驱。",
        "降低破坏主数值依赖的风险。",
        "drop edge e forbidden if e = argmax_j |a_ij|",
    ),
    "forbid_empty_rows": (
        "禁止删空前驱行",
        "为 1 时不允许把某一行所有 strict-lower 前驱全部删除。",
        "避免把原本有依赖的行变成无依赖行。",
        "pred_count_after[i] >= 1 if pred_count_before[i] >= 1",
    ),
    "dropped_nnz": (
        "删除非零元数",
        "被删除的 strict-lower 边数量，不含对角。",
        "衡量稀疏化代价。",
        "dropped_nnz = Σ_e 1[drop_mask[e]]",
    ),
    "dropped_rows": (
        "发生删除的行数",
        "至少删除了一个 strict-lower 前驱的行数。",
        "衡量删边分布是否集中。",
        "dropped_rows = Σ_i 1[row_drop_count[i] > 0]",
    ),
    "strict_lower_nnz_before": (
        "删除前严格下三角非零元数",
        "流式读取 L 的 j < i 边数量。",
        "删除预算和链统计的边数基线。",
        "strict_lower_nnz_before = |{(i,j): j < i}|",
    ),
    "strict_lower_nnz_after": (
        "删除后严格下三角非零元数",
        "删除 mask 后剩余的 strict-lower 边数量。",
        "检查实际稀疏化结果。",
        "strict_lower_nnz_after = strict_lower_nnz_before - dropped_nnz",
    ),
    "single_pred_rows_before": (
        "删除前单前驱行数",
        "删除前 pred_count[i] == 1 的行数。",
        "strict scan 链可用内部节点的原始规模。",
        "Σ_i 1[pred_count_before[i] == 1]",
    ),
    "single_pred_rows_after": (
        "删除后单前驱行数",
        "删除后 pred_count[i] == 1 的行数。",
        "判断删边是否把多前驱行转成可 scan 行。",
        "Σ_i 1[pred_count_after[i] == 1]",
    ),
    "multi_pred_rows_before": (
        "删除前多前驱行数",
        "删除前 pred_count[i] > 1 的行数。",
        "strict scan 被阻塞的主要来源。",
        "Σ_i 1[pred_count_before[i] > 1]",
    ),
    "multi_pred_rows_after": (
        "删除后多前驱行数",
        "删除后 pred_count[i] > 1 的行数。",
        "判断阻塞行是否减少。",
        "Σ_i 1[pred_count_after[i] > 1]",
    ),
    "multi_pred_to_single_pred_rows": (
        "多前驱转单前驱行数",
        "删除前 pred_count > 1 且删除后 pred_count == 1 的行数。",
        "scan-chain-aware 方法最直接想增加的行集合。",
        "Σ_i 1[pred_before[i] > 1 and pred_after[i] == 1]",
    ),
    "strict_chain_count_before": (
        "删除前严格链条数",
        "删除前按 strict single-predecessor 规则得到的长度 >= 2 链数。",
        "天然 strict scan 链数量。",
        "|{chain_before: len >= 2}|",
    ),
    "strict_chain_count_after": (
        "删除后严格链条数",
        "删除后按 strict single-predecessor 规则得到的长度 >= 2 链数。",
        "删元素后可执行 strict scan 链数量。",
        "|{chain_after: len >= 2}|",
    ),
    "strict_chain_row_coverage_before": (
        "删除前严格链覆盖行数",
        "删除前所有长度 >= 2 strict 链的链长之和。",
        "天然 strict scan 链覆盖规模。",
        "Σ len(chain_before), len >= 2",
    ),
    "strict_chain_row_coverage_after": (
        "删除后严格链覆盖行数",
        "删除后所有长度 >= 2 strict 链的链长之和。",
        "删元素后 strict scan 链覆盖规模。",
        "Σ len(chain_after), len >= 2",
    ),
    "strict_chain_row_coverage_gain": (
        "严格链覆盖行数增量",
        "删除后覆盖行数减删除前覆盖行数。",
        "直接衡量删边制造 scan 链的收益。",
        "strict_chain_row_coverage_after - strict_chain_row_coverage_before",
    ),
    "strict_chain_row_coverage_gain_ratio": (
        "严格链覆盖率增量",
        "严格链覆盖行数增量除以 n_rows。",
        "不同规模矩阵间可比的 strict 覆盖收益。",
        "strict_chain_row_coverage_gain / n_rows",
    ),
    "strict_chain_coverage_gain_per_drop_ratio": (
        "单位删除比例严格覆盖收益",
        "严格链覆盖率增量除以实际删除比例。",
        "回答每删除 1% 非零元大约换来多少 strict 覆盖率提升。",
        "strict_chain_row_coverage_gain_ratio / drop_ratio_actual",
    ),
    "strict_chain_coverage_gain_per_drop_percent": (
        "每删除1%严格覆盖率收益",
        "严格链覆盖率增量除以实际删除百分比点。",
        "直接回答每删除 1% strict-lower 非零元换来多少 strict 覆盖率提升。",
        "strict_chain_row_coverage_gain_ratio / (100 * drop_ratio_actual)",
    ),
    "frontier_safe_chain_row_coverage_before": (
        "删除前 frontier-safe 链覆盖行数",
        "删除前按唯一最深前驱加边界切链规则得到的链覆盖。",
        "比 strict 更宽松的可 scan 基线。",
        "Σ len(frontier_chain_before), len >= 2",
    ),
    "frontier_safe_chain_row_coverage_after": (
        "删除后 frontier-safe 链覆盖行数",
        "删除后按 frontier-safe 规则得到的链覆盖。",
        "判断删边是否改善 frontier-safe 候选链。",
        "Σ len(frontier_chain_after), len >= 2",
    ),
    "frontier_safe_chain_row_coverage_gain": (
        "frontier-safe 链覆盖行数增量",
        "删除后 frontier-safe 覆盖减删除前覆盖。",
        "衡量更宽松口径下的覆盖收益。",
        "after - before",
    ),
    "frontier_safe_chain_row_coverage_gain_ratio": (
        "frontier-safe 链覆盖率增量",
        "frontier-safe 覆盖行数增量除以 n_rows。",
        "不同矩阵间可比的 frontier-safe 覆盖收益。",
        "frontier_safe_chain_row_coverage_gain / n_rows",
    ),
    "frontier_safe_boundary_cut_before": (
        "删除前 frontier-safe 切链次数",
        "候选 parent 链中因非主依赖不满足 pred < head 发生的切链次数。",
        "frontier-safe 长链被打断的规模。",
        "# cuts before",
    ),
    "frontier_safe_boundary_cut_after": (
        "删除后 frontier-safe 切链次数",
        "删除后重新计算 frontier-safe 边界切链次数。",
        "判断删边是否减少安全边界切链。",
        "# cuts after",
    ),
    "frontier_safe_boundary_cut_reduction": (
        "frontier-safe 切链减少量",
        "删除前切链次数减删除后切链次数。",
        "scan-chain-aware 希望减少的 frontier-safe 切链点。",
        "frontier_safe_boundary_cut_before - frontier_safe_boundary_cut_after",
    ),
    "avg_chain_length_before": (
        "删除前平均严格链长",
        "删除前 strict 链覆盖行数除以 strict 链条数。",
        "天然 strict 链平均长度。",
        "coverage_before / chain_count_before",
    ),
    "avg_chain_length_after": (
        "删除后平均严格链长",
        "删除后 strict 链覆盖行数除以 strict 链条数。",
        "删边后 strict 链平均长度。",
        "coverage_after / chain_count_after",
    ),
    "max_chain_length_before": (
        "删除前最长严格链长",
        "删除前所有 strict 链中的最大链长。",
        "天然长链上限。",
        "max len(chain_before)",
    ),
    "max_chain_length_after": (
        "删除后最长严格链长",
        "删除后所有 strict 链中的最大链长。",
        "删边后长链上限。",
        "max len(chain_after)",
    ),
    "chain_gain_per_dropped_edge": (
        "每删一条边的严格覆盖收益",
        "严格链覆盖行数增量除以 dropped_nnz。",
        "衡量删边效率；负值表示删边破坏了 strict 链覆盖。",
        "strict_chain_row_coverage_gain / dropped_nnz",
    ),
    "effective_long_chain_score_before": (
        "删除前有效长链加权分数",
        "用长度 >=8、>=16、>=32 的覆盖行数加权求和。",
        "综合衡量删除前可执行长链规模，权重偏向更长链。",
        "score = rows_ge8 + 2 * rows_ge16 + 4 * rows_ge32",
    ),
    "effective_long_chain_score_after": (
        "删除后有效长链加权分数",
        "删除后重新计算长度 >=8、>=16、>=32 的覆盖行数加权和。",
        "综合衡量删边后可执行长链规模。",
        "score_after = rows_ge8_after + 2 * rows_ge16_after + 4 * rows_ge32_after",
    ),
    "effective_long_chain_score_gain": (
        "有效长链加权分数增量",
        "删除后有效长链加权分数减删除前分数。",
        "用于判断删边是否真正制造更有执行价值的长链。",
        "effective_long_chain_score_after - effective_long_chain_score_before",
    ),
    "effective_long_chain_score_gain_per_drop_percent": (
        "每删除1%有效长链分数收益",
        "有效长链加权分数增量先除以 n_rows 得到归一化收益，再除以实际删除百分比点。",
        "回答每删除 1% strict-lower 非零元换来多少综合长链收益。",
        "(effective_long_chain_score_gain / n_rows) / (100 * drop_ratio_actual)",
    ),
}

for _threshold in (8, 16, 32):
    DROP_FIELD_DEFINITIONS.update({
        f"effective_chain_rows_ge_{_threshold}_before": (
            f"删除前长度至少{_threshold}的有效链覆盖行数",
            f"删除前所有长度 >= {_threshold} 的 strict 链链长之和。",
            "衡量删除前已经有多少具备执行价值的长 strict 链行。",
            f"Σ len(chain_before), len >= {_threshold}",
        ),
        f"effective_chain_rows_ge_{_threshold}_after": (
            f"删除后长度至少{_threshold}的有效链覆盖行数",
            f"删除后所有长度 >= {_threshold} 的 strict 链链长之和。",
            "衡量删边后具备执行价值的长 strict 链行。",
            f"Σ len(chain_after), len >= {_threshold}",
        ),
        f"effective_chain_rows_ge_{_threshold}": (
            f"长度至少{_threshold}的有效链覆盖行数",
            f"兼容字段，等同于 effective_chain_rows_ge_{_threshold}_after。",
            "用于直接比较当前参数组合最终产生的有效长链覆盖。",
            f"effective_chain_rows_ge_{_threshold}_after",
        ),
        f"effective_chain_rows_ge_{_threshold}_gain": (
            f"长度至少{_threshold}的有效链覆盖行数增量",
            f"删除后长度 >= {_threshold} 的覆盖行数减删除前对应覆盖行数。",
            "判断删边是否真正制造更长、更值得执行的 scan 链。",
            (
                f"effective_chain_rows_ge_{_threshold}_after - "
                f"effective_chain_rows_ge_{_threshold}_before"
            ),
        ),
        f"effective_chain_rows_ge_{_threshold}_gain_per_dropped_edge": (
            f"每删一条边长度至少{_threshold}覆盖行数收益",
            f"长度 >= {_threshold} 的有效链覆盖行数增量除以 dropped_nnz。",
            "衡量每条被删除 strict-lower 边换来的有效长链行收益。",
            f"effective_chain_rows_ge_{_threshold}_gain / dropped_nnz",
        ),
        f"effective_chain_count_ge_{_threshold}_before": (
            f"删除前长度至少{_threshold}的有效链条数",
            f"删除前 strict 链中长度 >= {_threshold} 的链条数量。",
            "删除前具备执行价值的长链条数。",
            f"|{{chain_before: len >= {_threshold}}}|",
        ),
        f"effective_chain_count_ge_{_threshold}_after": (
            f"删除后长度至少{_threshold}的有效链条数",
            f"删除后 strict 链中长度 >= {_threshold} 的链条数量。",
            "删除后具备执行价值的长链条数。",
            f"|{{chain_after: len >= {_threshold}}}|",
        ),
        f"effective_chain_count_ge_{_threshold}_gain": (
            f"长度至少{_threshold}的有效链条数增量",
            f"删除后长度 >= {_threshold} 的链条数减删除前对应链条数。",
            "判断删边是否增加有执行价值的长链数量。",
            (
                f"effective_chain_count_ge_{_threshold}_after - "
                f"effective_chain_count_ge_{_threshold}_before"
            ),
        ),
        f"effective_chain_coverage_ge_{_threshold}_before": (
            f"删除前长度至少{_threshold}的有效链覆盖率",
            f"删除前长度 >= {_threshold} 的有效链覆盖行数除以 n_rows。",
            "不同矩阵间可比的删除前有效长链覆盖率。",
            f"effective_chain_rows_ge_{_threshold}_before / n_rows",
        ),
        f"effective_chain_coverage_ge_{_threshold}_after": (
            f"删除后长度至少{_threshold}的有效链覆盖率",
            f"删除后长度 >= {_threshold} 的有效链覆盖行数除以 n_rows。",
            "不同矩阵间可比的删除后有效长链覆盖率。",
            f"effective_chain_rows_ge_{_threshold}_after / n_rows",
        ),
        f"effective_chain_coverage_ge_{_threshold}": (
            f"长度至少{_threshold}的有效链覆盖率",
            f"兼容字段，等同于 effective_chain_coverage_ge_{_threshold}_after。",
            "用于画图和跨方法比较的最终有效长链覆盖率。",
            f"effective_chain_coverage_ge_{_threshold}_after",
        ),
        f"effective_chain_coverage_ge_{_threshold}_gain": (
            f"长度至少{_threshold}的有效链覆盖率增量",
            f"删除后长度 >= {_threshold} 的覆盖率减删除前对应覆盖率。",
            "判断删边是否提升有效长链覆盖率。",
            (
                f"effective_chain_coverage_ge_{_threshold}_after - "
                f"effective_chain_coverage_ge_{_threshold}_before"
            ),
        ),
        f"effective_chain_coverage_ge_{_threshold}_gain_per_drop_ratio": (
            f"单位删除比例长度至少{_threshold}覆盖率收益",
            f"长度 >= {_threshold} 的覆盖率增量除以实际删除比例。",
            "回答每删除 1% 非零元换来多少有效长链覆盖率提升。",
            (
                f"effective_chain_coverage_ge_{_threshold}_gain / "
                "drop_ratio_actual"
            ),
        ),
        f"effective_chain_coverage_ge_{_threshold}_gain_per_drop_percent": (
            f"每删除1%长度至少{_threshold}覆盖率收益",
            f"长度 >= {_threshold} 的覆盖率增量除以实际删除百分比点。",
            "直接回答每删除 1% strict-lower 非零元换来多少有效长链覆盖率提升。",
            (
                f"effective_chain_coverage_ge_{_threshold}_gain / "
                "(100 * drop_ratio_actual)"
            ),
        ),
        f"avg_effective_chain_length_ge_{_threshold}_before": (
            f"删除前长度至少{_threshold}的平均有效链长",
            f"删除前长度 >= {_threshold} 的有效链覆盖行数除以对应链条数。",
            "删除前长链的平均执行长度。",
            (
                f"effective_chain_rows_ge_{_threshold}_before / "
                f"effective_chain_count_ge_{_threshold}_before"
            ),
        ),
        f"avg_effective_chain_length_ge_{_threshold}_after": (
            f"删除后长度至少{_threshold}的平均有效链长",
            f"删除后长度 >= {_threshold} 的有效链覆盖行数除以对应链条数。",
            "删除后长链的平均执行长度。",
            (
                f"effective_chain_rows_ge_{_threshold}_after / "
                f"effective_chain_count_ge_{_threshold}_after"
            ),
        ),
        f"strict_chain_len_ge_{_threshold}_before": (
            f"删除前长度至少{_threshold}的严格链条数",
            f"删除前 strict 链中长度 >= {_threshold} 的链条数量。",
            "天然长 strict 链条数。",
            f"|{{chain_before: len >= {_threshold}}}|",
        ),
        f"strict_chain_len_ge_{_threshold}_after": (
            f"删除后长度至少{_threshold}的严格链条数",
            f"删除后 strict 链中长度 >= {_threshold} 的链条数量。",
            "删边后长 strict 链条数。",
            f"|{{chain_after: len >= {_threshold}}}|",
        ),
    })


@dataclass(frozen=True)
class ValuedDependencyGraph:
    n_rows: int
    strict_lower_nnz: int
    pred_count: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    abs_values: np.ndarray


@dataclass(frozen=True)
class ChainSummary:
    chain_count: int
    row_coverage: int
    avg_length: float
    max_length: int
    count_ge: dict[int, int]
    rows_ge: dict[int, int]


@dataclass(frozen=True)
class DropConfig:
    method: str
    drop_ratio_target: float
    risk_threshold: float | None
    max_drop_per_row: int
    keep_row_max_predecessor: bool
    forbid_empty_rows: bool


@dataclass(frozen=True)
class DropContext:
    dep_graph: base.StrictLowerDependencyGraph
    before_metrics: dict
    row_ids: np.ndarray
    pred_count: np.ndarray
    edge_risk: np.ndarray
    levels: np.ndarray
    frontier_parent: np.ndarray
    frontier_parent_edge_mask: np.ndarray
    long_chain_keep_edge: np.ndarray
    row_long_chain_potential: np.ndarray
    row_max_edge_mask: np.ndarray


def _entry_abs_value(parts, field: str):
    if field == "pattern":
        return 1.0
    if field == "integer":
        return abs(float(int(parts[2])))
    if field == "real":
        return abs(float(parts[2]))
    if field == "complex":
        return math.hypot(float(parts[2]), float(parts[3]))
    raise ValueError(f"Unsupported Matrix Market field: {field}")


def read_valued_strict_lower_graph_mtx(path: Path, triangle: str = "lower"):
    deps = base.read_strict_lower_dependencies_mtx(path, triangle=triangle)
    index_dtype = base._index_dtype(deps.n_rows)
    ptr_dtype = base._count_dtype(deps.strict_lower_nnz)

    indptr = np.zeros(deps.n_rows + 1, dtype=ptr_dtype)
    indptr[1:] = np.cumsum(deps.pred_count, dtype=ptr_dtype)
    indices = np.empty(deps.strict_lower_nnz, dtype=index_dtype)
    abs_values = np.empty(deps.strict_lower_nnz, dtype=np.float64)
    cursor = indptr[:-1].copy()

    with path.open("r", encoding="utf-8", errors="replace") as f:
        banner = f.readline()
        if not banner:
            raise ValueError(f"Empty Matrix Market file: {path.name}")

        field, symmetry = base._parse_matrix_market_banner(banner, path)
        n_rows, n_cols, reported_nnz, line_number = base._read_matrix_market_size(
            f, path, start_line_number=1
        )
        header = base.MatrixMarketHeader(
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

            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid Matrix Market entry at {path.name}:{line_number}"
                )

            row = int(parts[0]) - 1
            col = int(parts[1]) - 1
            abs_value = _entry_abs_value(parts, field)

            if row > col:
                write_at = int(cursor[row])
                indices[write_at] = col
                abs_values[write_at] = abs_value
                cursor[row] += 1
            elif expands_symmetric and col > row:
                write_at = int(cursor[col])
                indices[write_at] = row
                abs_values[write_at] = abs_value
                cursor[col] += 1

    if not np.array_equal(cursor, indptr[1:]):
        raise RuntimeError(f"Failed to fill valued dependency graph: {path.name}")

    return ValuedDependencyGraph(
        n_rows=deps.n_rows,
        strict_lower_nnz=deps.strict_lower_nnz,
        pred_count=deps.pred_count.copy(),
        indptr=indptr,
        indices=indices,
        abs_values=abs_values,
    )


def valued_to_dependency_graph(graph: ValuedDependencyGraph):
    pred_count = graph.pred_count.astype(
        base._count_dtype(graph.strict_lower_nnz),
        copy=False,
    )
    first_pred = np.full(graph.n_rows, -1, dtype=graph.indices.dtype)
    nonempty_rows = np.flatnonzero(graph.pred_count > 0)
    if nonempty_rows.size:
        first_pred[nonempty_rows] = graph.indices[graph.indptr[nonempty_rows]]

    return base.StrictLowerDependencyGraph(
        n_rows=graph.n_rows,
        strict_lower_nnz=graph.strict_lower_nnz,
        pred_count=pred_count,
        first_pred=first_pred,
        indptr=graph.indptr,
        indices=graph.indices,
    )


def _row_ids(graph: ValuedDependencyGraph):
    counts = graph.pred_count.astype(np.int64, copy=False)
    return np.repeat(np.arange(graph.n_rows, dtype=graph.indices.dtype), counts)


def _edge_risk_and_row_max_mask(graph: ValuedDependencyGraph):
    if graph.strict_lower_nnz == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=bool),
        )

    nonempty_rows = np.flatnonzero(graph.pred_count > 0)
    row_starts = graph.indptr[nonempty_rows].astype(np.int64, copy=False)
    row_ends = graph.indptr[nonempty_rows + 1].astype(np.int64, copy=False)
    row_max = np.maximum.reduceat(graph.abs_values, row_starts)
    row_max_repeated = np.repeat(row_max, graph.pred_count[graph.pred_count > 0])
    edge_risk = (
        graph.abs_values / np.maximum(row_max_repeated, 1e-300)
    ).astype(np.float32, copy=False)

    row_max_mask = np.zeros(graph.strict_lower_nnz, dtype=bool)
    for start, end, max_value in zip(row_starts, row_ends, row_max, strict=False):
        row_max_mask[start:end] = graph.abs_values[start:end] == max_value
    return edge_risk, row_max_mask


def strict_chain_summary(dep_graph: base.StrictLowerDependencyGraph):
    pred_count = dep_graph.pred_count.copy()
    first_pred = dep_graph.first_pred.copy()
    strict_parent = base.build_strict_parent(pred_count, first_pred)
    children = base.build_strict_child_arrays(strict_parent)

    thresholds = (2, 4, 8, 16, 32, 64)
    count_ge = {threshold: 0 for threshold in thresholds}
    rows_ge = {threshold: 0 for threshold in thresholds}
    chain_count = 0
    row_coverage = 0
    max_length = 0

    for head in range(dep_graph.n_rows):
        if not base._is_strict_boundary_head(
            head, pred_count, strict_parent, children.strict_child_count
        ):
            continue

        length = 1
        current = head
        while children.strict_child_count[current] == 1:
            child = int(children.unique_strict_child[current])
            if child == -1:
                raise RuntimeError(f"Missing unique strict child for row {current}")
            current = child
            length += 1

        if length < 2:
            continue

        chain_count += 1
        row_coverage += length
        max_length = max(max_length, length)
        for threshold in thresholds:
            if length >= threshold:
                count_ge[threshold] += 1
                rows_ge[threshold] += length

    avg_length = row_coverage / chain_count if chain_count else 0.0
    return ChainSummary(
        chain_count=chain_count,
        row_coverage=row_coverage,
        avg_length=float(avg_length),
        max_length=int(max_length),
        count_ge=count_ge,
        rows_ge=rows_ge,
    )


def frontier_stats(dep_graph: base.StrictLowerDependencyGraph):
    stats, _ = base.analyze_frontier_safe_chains_from_graph(dep_graph, dump_chains=False)
    return stats


def graph_after_drop(
    graph: ValuedDependencyGraph,
    drop_mask: np.ndarray,
    row_ids: np.ndarray | None = None,
):
    keep_mask = ~drop_mask
    if row_ids is None:
        row_ids = _row_ids(graph)
    keep_counts = np.bincount(
        row_ids[keep_mask].astype(np.int64, copy=False),
        minlength=graph.n_rows,
    )
    ptr_dtype = base._count_dtype(int(np.count_nonzero(keep_mask)))
    indptr = np.zeros(graph.n_rows + 1, dtype=ptr_dtype)
    indptr[1:] = np.cumsum(keep_counts, dtype=ptr_dtype)

    return ValuedDependencyGraph(
        n_rows=graph.n_rows,
        strict_lower_nnz=int(np.count_nonzero(keep_mask)),
        pred_count=keep_counts.astype(base._count_dtype(int(np.count_nonzero(keep_mask))), copy=False),
        indptr=indptr,
        indices=graph.indices[keep_mask].copy(),
        abs_values=graph.abs_values[keep_mask].copy(),
    )


def _frontier_parent_edge_mask(dep_graph: base.StrictLowerDependencyGraph, frontier_parent):
    parent_mask = np.zeros(dep_graph.strict_lower_nnz, dtype=bool)
    if frontier_parent is None:
        _, frontier_parent, _ = base.build_frontier_safe_parent(dep_graph)
    for row in range(dep_graph.n_rows):
        parent = int(frontier_parent[row])
        if parent == -1:
            continue
        start = int(dep_graph.indptr[row])
        end = int(dep_graph.indptr[row + 1])
        for edge in range(start, end):
            if int(dep_graph.indices[edge]) == parent:
                parent_mask[edge] = True
                break
    return parent_mask


def _edge_index_dtype(n_edges: int):
    if n_edges <= np.iinfo(np.int32).max:
        return np.int32
    return np.int64


def _strict_local_lengths(dep_graph: base.StrictLowerDependencyGraph):
    pred_count = dep_graph.pred_count.copy()
    first_pred = dep_graph.first_pred.copy()
    strict_parent = base.build_strict_parent(pred_count, first_pred)
    children = base.build_strict_child_arrays(strict_parent)

    upstream_len = np.ones(dep_graph.n_rows, dtype=np.int32)
    for row in range(dep_graph.n_rows):
        parent = int(strict_parent[row])
        if parent != -1 and children.strict_child_count[parent] == 1:
            upstream_len[row] = upstream_len[parent] + 1

    downstream_len = np.ones(dep_graph.n_rows, dtype=np.int32)
    for row in range(dep_graph.n_rows - 1, -1, -1):
        if children.strict_child_count[row] != 1:
            continue
        child = int(children.unique_strict_child[row])
        if child == -1:
            raise RuntimeError(f"Missing unique strict child for row {row}")
        downstream_len[row] = downstream_len[child] + 1

    return children.strict_child_count, upstream_len, downstream_len


def _long_chain_keep_edges(
    graph: ValuedDependencyGraph,
    levels: np.ndarray,
    strict_child_count: np.ndarray,
    upstream_len: np.ndarray,
    downstream_len: np.ndarray,
):
    keep_edge = np.full(
        graph.n_rows,
        -1,
        dtype=_edge_index_dtype(graph.strict_lower_nnz),
    )
    row_potential = np.ones(graph.n_rows, dtype=np.int32)

    for row in range(graph.n_rows):
        start = int(graph.indptr[row])
        end = int(graph.indptr[row + 1])
        if start == end:
            continue

        best_edge = start
        best_potential = -1
        best_level = -1
        best_abs = -1.0
        downstream = int(downstream_len[row])
        for edge in range(start, end):
            pred = int(graph.indices[edge])
            upstream = int(upstream_len[pred]) if strict_child_count[pred] == 0 else 0
            potential = upstream + downstream
            pred_level = int(levels[pred])
            abs_value = float(graph.abs_values[edge])
            if (
                potential > best_potential
                or (
                    potential == best_potential
                    and (
                        pred_level > best_level
                        or (pred_level == best_level and abs_value > best_abs)
                    )
                )
            ):
                best_edge = edge
                best_potential = potential
                best_level = pred_level
                best_abs = abs_value

        keep_edge[row] = best_edge
        row_potential[row] = max(best_potential, 1)

    return keep_edge, row_potential


def _long_chain_value(lengths):
    values = np.asarray(lengths, dtype=np.float32)
    return (
        np.where(values >= 8, values, 0.0)
        + np.where(values >= 16, 2.0 * values, 0.0)
        + np.where(values >= 32, 4.0 * values, 0.0)
    )


def metrics_for_dep_graph(dep_graph: base.StrictLowerDependencyGraph):
    strict_summary = strict_chain_summary(dep_graph)
    frontier = frontier_stats(dep_graph)
    pred_count = dep_graph.pred_count
    return {
        "dep_graph": dep_graph,
        "strict": strict_summary,
        "frontier": frontier,
        "single_pred_rows": int(np.count_nonzero(pred_count == 1)),
        "multi_pred_rows": int(np.count_nonzero(pred_count > 1)),
    }


def metrics_for_graph(graph: ValuedDependencyGraph):
    return metrics_for_dep_graph(valued_to_dependency_graph(graph))


def _avg_effective_chain_length(summary: ChainSummary, threshold: int):
    count = summary.count_ge[threshold]
    return (summary.rows_ge[threshold] / count) if count else 0.0


def _effective_long_chain_score(summary: ChainSummary):
    return (
        summary.rows_ge[8]
        + 2 * summary.rows_ge[16]
        + 4 * summary.rows_ge[32]
    )


def build_drop_context(graph: ValuedDependencyGraph):
    dep_graph = valued_to_dependency_graph(graph)
    before_metrics = metrics_for_dep_graph(dep_graph)
    row_ids = _row_ids(graph)
    edge_risk, row_max_edge_mask = _edge_risk_and_row_max_mask(graph)
    levels, frontier_parent, _ = base.build_frontier_safe_parent(dep_graph)
    frontier_parent_edge_mask = _frontier_parent_edge_mask(dep_graph, frontier_parent)
    strict_child_count, upstream_len, downstream_len = _strict_local_lengths(dep_graph)
    long_chain_keep_edge, row_long_chain_potential = _long_chain_keep_edges(
        graph=graph,
        levels=levels,
        strict_child_count=strict_child_count,
        upstream_len=upstream_len,
        downstream_len=downstream_len,
    )
    return DropContext(
        dep_graph=dep_graph,
        before_metrics=before_metrics,
        row_ids=row_ids,
        pred_count=graph.pred_count,
        edge_risk=edge_risk,
        levels=levels,
        frontier_parent=frontier_parent,
        frontier_parent_edge_mask=frontier_parent_edge_mask,
        long_chain_keep_edge=long_chain_keep_edge,
        row_long_chain_potential=row_long_chain_potential,
        row_max_edge_mask=row_max_edge_mask,
    )


def select_drop_edges(
    graph: ValuedDependencyGraph,
    config: DropConfig,
    context: DropContext | None = None,
):
    if config.method == "no-drop" or config.drop_ratio_target <= 0:
        return np.zeros(graph.strict_lower_nnz, dtype=bool)

    target_drop_count = int(math.floor(graph.strict_lower_nnz * config.drop_ratio_target))
    if target_drop_count <= 0:
        return np.zeros(graph.strict_lower_nnz, dtype=bool)

    if context is None:
        context = build_drop_context(graph)

    row_ids = context.row_ids
    pred_count = context.pred_count
    edge_risk = context.edge_risk
    levels = context.levels
    frontier_parent = context.frontier_parent
    frontier_parent_edge_mask = context.frontier_parent_edge_mask
    long_chain_keep_edge = context.long_chain_keep_edge
    row_long_chain_potential = context.row_long_chain_potential
    row_max_edge_mask = context.row_max_edge_mask

    candidate_mask = np.ones(graph.strict_lower_nnz, dtype=bool)
    if config.risk_threshold is not None:
        candidate_mask &= edge_risk <= config.risk_threshold
    if config.keep_row_max_predecessor:
        candidate_mask &= ~row_max_edge_mask
    if config.method in {"scan-chain-aware", "long-chain-aware-drop"}:
        protected = long_chain_keep_edge[long_chain_keep_edge >= 0]
        candidate_mask[protected] = False
        candidate_mask &= pred_count[row_ids] > 1
        if config.max_drop_per_row >= 0:
            candidate_mask &= (pred_count[row_ids] - 1) <= config.max_drop_per_row
    if config.method == "long-chain-aware-drop":
        candidate_mask &= row_long_chain_potential[row_ids] >= 8

    edge_indices = np.flatnonzero(candidate_mask)
    if edge_indices.size == 0:
        return np.zeros(graph.strict_lower_nnz, dtype=bool)

    candidate_rows = row_ids[edge_indices]
    candidate_risks = edge_risk[edge_indices]
    max_level = max(int(np.max(levels)) if graph.n_rows else 1, 1)

    if config.method == "magnitude-drop":
        score = -candidate_risks.astype(np.float32, copy=False)
    elif config.method == "wavefront-oriented":
        candidate_preds = graph.indices[edge_indices]
        pred_levels = levels[candidate_preds].astype(np.float32, copy=False)
        is_candidate_parent = frontier_parent_edge_mask[edge_indices]
        score = (
            (pred_levels / max_level)
            + (0.35 * is_candidate_parent.astype(np.float32, copy=False))
            - candidate_risks
        )
    elif config.method == "scan-chain-aware":
        candidate_preds = graph.indices[edge_indices]
        degrees = pred_count[candidate_rows]
        needed_drops = np.maximum(degrees.astype(np.float32) - 1.0, 1.0)
        potential = row_long_chain_potential[candidate_rows].astype(
            np.float32,
            copy=False,
        )
        long_value = _long_chain_value(potential)
        conversion_bonus = np.where(
            degrees == 2,
            3.0,
            1.0 / needed_drops,
        ).astype(np.float32, copy=False)
        non_main_bonus = (
            frontier_parent[candidate_rows] != candidate_preds
        ).astype(np.float32, copy=False)
        level_bonus = levels[candidate_rows].astype(np.float32, copy=False) / max_level
        short_penalty = np.where(potential < 8.0, 1.5 * (8.0 - potential), 0.0)
        score = (
            2.0 * conversion_bonus
            + 0.20 * (long_value / needed_drops)
            + 1.2 * non_main_bonus
            + 0.3 * level_bonus
            + 2.0 * (1.0 - candidate_risks)
            - short_penalty
        )
    elif config.method == "long-chain-aware-drop":
        degrees = pred_count[candidate_rows]
        needed_drops = np.maximum(degrees.astype(np.float32) - 1.0, 1.0)
        potential = row_long_chain_potential[candidate_rows].astype(
            np.float32,
            copy=False,
        )
        long_value = _long_chain_value(potential)
        non_main_bonus = (~frontier_parent_edge_mask[edge_indices]).astype(
            np.float32,
            copy=False,
        )
        level_bonus = levels[candidate_rows].astype(np.float32, copy=False) / max_level
        score = (
            (long_value / needed_drops)
            + 0.75 * potential
            + 3.0 * non_main_bonus
            + 0.5 * level_bonus
            - 6.0 * candidate_risks
        )
    else:
        raise ValueError(f"Unsupported drop method: {config.method}")

    order = np.lexsort((edge_indices, candidate_risks, -score))

    drop_mask = np.zeros(graph.strict_lower_nnz, dtype=bool)
    row_drop_count = np.zeros(graph.n_rows, dtype=np.int32)
    selected = 0
    for ordered_index in order:
        edge = int(edge_indices[ordered_index])
        row = int(row_ids[edge])
        if selected >= target_drop_count:
            break
        if drop_mask[edge]:
            continue
        if config.max_drop_per_row >= 0 and row_drop_count[row] >= config.max_drop_per_row:
            continue
        remaining_after = pred_count[row] - int(row_drop_count[row]) - 1
        if config.forbid_empty_rows and remaining_after <= 0:
            continue

        drop_mask[edge] = True
        row_drop_count[row] += 1
        selected += 1

    return drop_mask


def drop_experiment_row(
    matrix_name: str,
    source: str,
    input_path: Path,
    before_graph: ValuedDependencyGraph,
    context: DropContext,
    config: DropConfig,
):
    before_metrics = context.before_metrics
    drop_mask = select_drop_edges(before_graph, config, context=context)
    if np.any(drop_mask):
        after_graph = graph_after_drop(before_graph, drop_mask, context.row_ids)
        after_metrics = metrics_for_graph(after_graph)
    else:
        after_graph = before_graph
        after_metrics = before_metrics

    row_ids = context.row_ids
    dropped_nnz = int(np.count_nonzero(drop_mask))
    if dropped_nnz:
        dropped_rows = int(np.unique(row_ids[drop_mask]).size)
    else:
        dropped_rows = 0

    before_pred_count = before_metrics["dep_graph"].pred_count
    after_pred_count = after_metrics["dep_graph"].pred_count
    multi_to_single = int(np.count_nonzero((before_pred_count > 1) & (after_pred_count == 1)))

    before_strict = before_metrics["strict"]
    after_strict = after_metrics["strict"]
    before_frontier = before_metrics["frontier"]
    after_frontier = after_metrics["frontier"]

    strict_gain = after_strict.row_coverage - before_strict.row_coverage
    frontier_gain = (
        after_frontier["frontier_safe_chain_row_coverage"]
        - before_frontier["frontier_safe_chain_row_coverage"]
    )
    boundary_cut_reduction = (
        before_frontier["frontier_safe_boundary_cut_rows"]
        - after_frontier["frontier_safe_boundary_cut_rows"]
    )
    drop_ratio_actual = (
        dropped_nnz / before_graph.strict_lower_nnz
        if before_graph.strict_lower_nnz
        else 0.0
    )
    strict_gain_ratio = (
        strict_gain / before_graph.n_rows if before_graph.n_rows else 0.0
    )
    frontier_gain_ratio = (
        frontier_gain / before_graph.n_rows if before_graph.n_rows else 0.0
    )
    effective_rows_before = {
        threshold: before_strict.rows_ge[threshold]
        for threshold in (8, 16, 32)
    }
    effective_rows_after = {
        threshold: after_strict.rows_ge[threshold]
        for threshold in (8, 16, 32)
    }
    effective_row_gain = {
        threshold: effective_rows_after[threshold] - effective_rows_before[threshold]
        for threshold in (8, 16, 32)
    }
    effective_count_before = {
        threshold: before_strict.count_ge[threshold]
        for threshold in (8, 16, 32)
    }
    effective_count_after = {
        threshold: after_strict.count_ge[threshold]
        for threshold in (8, 16, 32)
    }
    effective_count_gain = {
        threshold: effective_count_after[threshold] - effective_count_before[threshold]
        for threshold in (8, 16, 32)
    }
    effective_coverage_before = {
        threshold: (
            effective_rows_before[threshold] / before_graph.n_rows
            if before_graph.n_rows
            else 0.0
        )
        for threshold in (8, 16, 32)
    }
    effective_coverage_after = {
        threshold: (
            effective_rows_after[threshold] / before_graph.n_rows
            if before_graph.n_rows
            else 0.0
        )
        for threshold in (8, 16, 32)
    }
    effective_coverage_gain = {
        threshold: effective_coverage_after[threshold] - effective_coverage_before[threshold]
        for threshold in (8, 16, 32)
    }
    avg_effective_before = {
        threshold: _avg_effective_chain_length(before_strict, threshold)
        for threshold in (8, 16, 32)
    }
    avg_effective_after = {
        threshold: _avg_effective_chain_length(after_strict, threshold)
        for threshold in (8, 16, 32)
    }
    long_score_before = _effective_long_chain_score(before_strict)
    long_score_after = _effective_long_chain_score(after_strict)
    long_score_gain = long_score_after - long_score_before
    long_score_gain_ratio = (
        long_score_gain / before_graph.n_rows if before_graph.n_rows else 0.0
    )

    return {
        "status": "ok",
        "error": "",
        "matrix_name": matrix_name,
        "source": source,
        "input_path": str(input_path),
        "n_rows": before_graph.n_rows,
        "drop_method": config.method,
        "drop_ratio_target": config.drop_ratio_target,
        "drop_ratio_actual": drop_ratio_actual,
        "risk_threshold": "" if config.risk_threshold is None else config.risk_threshold,
        "max_drop_per_row": config.max_drop_per_row,
        "keep_row_max_predecessor": int(config.keep_row_max_predecessor),
        "forbid_empty_rows": int(config.forbid_empty_rows),
        "dropped_nnz": dropped_nnz,
        "dropped_rows": dropped_rows,
        "strict_lower_nnz_before": before_graph.strict_lower_nnz,
        "strict_lower_nnz_after": after_graph.strict_lower_nnz,
        "single_pred_rows_before": before_metrics["single_pred_rows"],
        "single_pred_rows_after": after_metrics["single_pred_rows"],
        "multi_pred_rows_before": before_metrics["multi_pred_rows"],
        "multi_pred_rows_after": after_metrics["multi_pred_rows"],
        "multi_pred_to_single_pred_rows": multi_to_single,
        "strict_chain_count_before": before_strict.chain_count,
        "strict_chain_count_after": after_strict.chain_count,
        "strict_chain_row_coverage_before": before_strict.row_coverage,
        "strict_chain_row_coverage_after": after_strict.row_coverage,
        "strict_chain_row_coverage_gain": strict_gain,
        "strict_chain_row_coverage_gain_ratio": strict_gain_ratio,
        "strict_chain_coverage_gain_per_drop_ratio": (
            strict_gain_ratio / drop_ratio_actual if drop_ratio_actual else 0.0
        ),
        "strict_chain_coverage_gain_per_drop_percent": (
            strict_gain_ratio / (100.0 * drop_ratio_actual)
            if drop_ratio_actual
            else 0.0
        ),
        "frontier_safe_chain_row_coverage_before": before_frontier["frontier_safe_chain_row_coverage"],
        "frontier_safe_chain_row_coverage_after": after_frontier["frontier_safe_chain_row_coverage"],
        "frontier_safe_chain_row_coverage_gain": frontier_gain,
        "frontier_safe_chain_row_coverage_gain_ratio": frontier_gain_ratio,
        "frontier_safe_boundary_cut_before": before_frontier["frontier_safe_boundary_cut_rows"],
        "frontier_safe_boundary_cut_after": after_frontier["frontier_safe_boundary_cut_rows"],
        "frontier_safe_boundary_cut_reduction": boundary_cut_reduction,
        "effective_chain_rows_ge_8_before": effective_rows_before[8],
        "effective_chain_rows_ge_8_after": effective_rows_after[8],
        "effective_chain_rows_ge_8": effective_rows_after[8],
        "effective_chain_rows_ge_8_gain": effective_row_gain[8],
        "effective_chain_rows_ge_8_gain_per_dropped_edge": (
            effective_row_gain[8] / dropped_nnz if dropped_nnz else 0.0
        ),
        "effective_chain_rows_ge_16_before": effective_rows_before[16],
        "effective_chain_rows_ge_16_after": effective_rows_after[16],
        "effective_chain_rows_ge_16": effective_rows_after[16],
        "effective_chain_rows_ge_16_gain": effective_row_gain[16],
        "effective_chain_rows_ge_16_gain_per_dropped_edge": (
            effective_row_gain[16] / dropped_nnz if dropped_nnz else 0.0
        ),
        "effective_chain_rows_ge_32_before": effective_rows_before[32],
        "effective_chain_rows_ge_32_after": effective_rows_after[32],
        "effective_chain_rows_ge_32": effective_rows_after[32],
        "effective_chain_rows_ge_32_gain": effective_row_gain[32],
        "effective_chain_rows_ge_32_gain_per_dropped_edge": (
            effective_row_gain[32] / dropped_nnz if dropped_nnz else 0.0
        ),
        "effective_chain_count_ge_8_before": effective_count_before[8],
        "effective_chain_count_ge_8_after": effective_count_after[8],
        "effective_chain_count_ge_8_gain": effective_count_gain[8],
        "effective_chain_count_ge_16_before": effective_count_before[16],
        "effective_chain_count_ge_16_after": effective_count_after[16],
        "effective_chain_count_ge_16_gain": effective_count_gain[16],
        "effective_chain_count_ge_32_before": effective_count_before[32],
        "effective_chain_count_ge_32_after": effective_count_after[32],
        "effective_chain_count_ge_32_gain": effective_count_gain[32],
        "effective_chain_coverage_ge_8_before": effective_coverage_before[8],
        "effective_chain_coverage_ge_8_after": effective_coverage_after[8],
        "effective_chain_coverage_ge_8": effective_coverage_after[8],
        "effective_chain_coverage_ge_8_gain": effective_coverage_gain[8],
        "effective_chain_coverage_ge_8_gain_per_drop_ratio": (
            effective_coverage_gain[8] / drop_ratio_actual
            if drop_ratio_actual
            else 0.0
        ),
        "effective_chain_coverage_ge_8_gain_per_drop_percent": (
            effective_coverage_gain[8] / (100.0 * drop_ratio_actual)
            if drop_ratio_actual
            else 0.0
        ),
        "effective_chain_coverage_ge_16_before": effective_coverage_before[16],
        "effective_chain_coverage_ge_16_after": effective_coverage_after[16],
        "effective_chain_coverage_ge_16": effective_coverage_after[16],
        "effective_chain_coverage_ge_16_gain": effective_coverage_gain[16],
        "effective_chain_coverage_ge_16_gain_per_drop_ratio": (
            effective_coverage_gain[16] / drop_ratio_actual
            if drop_ratio_actual
            else 0.0
        ),
        "effective_chain_coverage_ge_16_gain_per_drop_percent": (
            effective_coverage_gain[16] / (100.0 * drop_ratio_actual)
            if drop_ratio_actual
            else 0.0
        ),
        "effective_chain_coverage_ge_32_before": effective_coverage_before[32],
        "effective_chain_coverage_ge_32_after": effective_coverage_after[32],
        "effective_chain_coverage_ge_32": effective_coverage_after[32],
        "effective_chain_coverage_ge_32_gain": effective_coverage_gain[32],
        "effective_chain_coverage_ge_32_gain_per_drop_ratio": (
            effective_coverage_gain[32] / drop_ratio_actual
            if drop_ratio_actual
            else 0.0
        ),
        "effective_chain_coverage_ge_32_gain_per_drop_percent": (
            effective_coverage_gain[32] / (100.0 * drop_ratio_actual)
            if drop_ratio_actual
            else 0.0
        ),
        "avg_chain_length_before": before_strict.avg_length,
        "avg_chain_length_after": after_strict.avg_length,
        "avg_effective_chain_length_ge_8_before": avg_effective_before[8],
        "avg_effective_chain_length_ge_8_after": avg_effective_after[8],
        "avg_effective_chain_length_ge_16_before": avg_effective_before[16],
        "avg_effective_chain_length_ge_16_after": avg_effective_after[16],
        "avg_effective_chain_length_ge_32_before": avg_effective_before[32],
        "avg_effective_chain_length_ge_32_after": avg_effective_after[32],
        "effective_long_chain_score_before": long_score_before,
        "effective_long_chain_score_after": long_score_after,
        "effective_long_chain_score_gain": long_score_gain,
        "effective_long_chain_score_gain_per_drop_percent": (
            long_score_gain_ratio / (100.0 * drop_ratio_actual)
            if drop_ratio_actual
            else 0.0
        ),
        "max_chain_length_before": before_strict.max_length,
        "max_chain_length_after": after_strict.max_length,
        "strict_chain_len_ge_8_before": before_strict.count_ge[8],
        "strict_chain_len_ge_8_after": after_strict.count_ge[8],
        "strict_chain_len_ge_16_before": before_strict.count_ge[16],
        "strict_chain_len_ge_16_after": after_strict.count_ge[16],
        "strict_chain_len_ge_32_before": before_strict.count_ge[32],
        "strict_chain_len_ge_32_after": after_strict.count_ge[32],
        "chain_gain_per_dropped_edge": strict_gain / dropped_nnz if dropped_nnz else 0.0,
    }


def parse_float_list(value: str):
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_int_list(value: str):
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_method_list(value: str):
    methods = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [method for method in methods if method not in DROP_METHODS]
    if unknown:
        raise ValueError(f"Unsupported drop methods: {unknown}")
    return methods


def iter_drop_configs(
    methods,
    drop_ratios,
    risk_thresholds,
    max_drop_per_row_values,
    keep_row_max_predecessor,
    forbid_empty_rows,
):
    if "no-drop" in methods:
        yield DropConfig(
            method="no-drop",
            drop_ratio_target=0.0,
            risk_threshold=None,
            max_drop_per_row=0,
            keep_row_max_predecessor=keep_row_max_predecessor,
            forbid_empty_rows=forbid_empty_rows,
        )

    for method in methods:
        if method == "no-drop":
            continue
        for ratio in drop_ratios:
            for risk_threshold in risk_thresholds:
                for max_drop_per_row in max_drop_per_row_values:
                    yield DropConfig(
                        method=method,
                        drop_ratio_target=ratio,
                        risk_threshold=risk_threshold,
                        max_drop_per_row=max_drop_per_row,
                        keep_row_max_predecessor=keep_row_max_predecessor,
                        forbid_empty_rows=forbid_empty_rows,
                    )


def collect_inputs(input_path: Path, source: str, limit: int | None):
    files = base.collect_mtx_files(input_path, source=source)
    if limit is not None:
        return files[:limit]
    return files


def resolve_experiment_input(input_arg: str | None, source: str):
    if input_arg:
        return base.resolve_input_path(input_arg)
    if source == "factor-l":
        return base.SPCG_FACTOR_DIR
    return base.REAL_MATRIX_DIR


def resolve_matrix_for_spcg(path: Path):
    return base.resolve_input_path(str(path))


def run_experiment(args):
    methods = parse_method_list(args.methods)
    drop_ratios = parse_float_list(args.drop_ratios)
    risk_thresholds = parse_float_list(args.risk_thresholds)
    max_drop_per_row_values = parse_int_list(args.max_drop_per_row)
    input_path = resolve_experiment_input(args.input, args.source)
    output_path = base.resolve_repo_relative_path(args.out)
    summary_path = base.resolve_repo_relative_path(args.summary_out)
    fig_dir = base.resolve_repo_relative_path(args.fig_dir)
    factor_dir = base.resolve_repo_relative_path(args.factor_dir)
    definitions_path = base.resolve_repo_relative_path(args.definitions_out)

    configs = list(iter_drop_configs(
        methods=methods,
        drop_ratios=drop_ratios,
        risk_thresholds=risk_thresholds,
        max_drop_per_row_values=max_drop_per_row_values,
        keep_row_max_predecessor=not args.allow_drop_row_max_predecessor,
        forbid_empty_rows=not args.allow_empty_rows,
    ))

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    rows = []
    files = collect_inputs(input_path, source=args.source, limit=args.limit)
    for matrix_path in files:
        try:
            matrix_name = matrix_path.stem
            analysis_path = matrix_path
            if args.source == "spcg-ilu-l":
                l_path, _, matrix_name = base.generate_spcg_lu_factors(
                    matrix_path=matrix_path,
                    factor_dir=factor_dir,
                    method=args.spcg_method,
                    fill_factor=args.fill_factor,
                    drop_tol=args.drop_tol,
                    sparsify_percentage=args.spcg_sparsify_percentage,
                )
                analysis_path = l_path

            graph = read_valued_strict_lower_graph_mtx(analysis_path)
            context = build_drop_context(graph)
            for config in configs:
                rows.append(drop_experiment_row(
                    matrix_name=matrix_name,
                    source=args.source,
                    input_path=analysis_path,
                    before_graph=graph,
                    context=context,
                    config=config,
                ))
        except Exception as exc:
            rows.append({
                "status": "failed",
                "error": str(exc),
                "matrix_name": matrix_path.stem,
                "source": args.source,
                "input_path": str(matrix_path),
            })
            print(f"[WARN] Skip {matrix_path.name}: {exc}")

    all_rows = write_drop_csv(
        output_path,
        rows,
        replace=args.replace_out,
    )
    write_summary(summary_path, all_rows)
    write_field_definitions_csv(definitions_path)
    if not args.no_plots:
        write_plots(fig_dir, all_rows)

    return output_path, summary_path, fig_dir, definitions_path, all_rows


def _drop_row_key(row):
    return tuple(str(row.get(field, "")) for field in DROP_UPSERT_KEY_FIELDS)


def _read_existing_drop_rows(output_path: Path):
    if not output_path.exists() or output_path.stat().st_size == 0:
        return []

    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != DROP_CSV_FIELDS:
            raise ValueError(
                f"Existing CSV schema does not match current fields: {output_path}"
            )
        return list(reader)


def write_drop_csv(output_path: Path, rows, replace=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        merged_rows = list(rows)
    else:
        merged_by_key = {
            _drop_row_key(row): row
            for row in _read_existing_drop_rows(output_path)
        }
        row_order = list(merged_by_key)
        for row in rows:
            key = _drop_row_key(row)
            if key not in merged_by_key:
                row_order.append(key)
            merged_by_key[key] = row
        merged_rows = [merged_by_key[key] for key in row_order]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DROP_CSV_FIELDS)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({field: row.get(field, "") for field in DROP_CSV_FIELDS})
    return merged_rows


def write_field_definitions_csv(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "parameter",
                "chinese_name",
                "calculation_description",
                "meaning",
                "formula",
            ],
        )
        writer.writeheader()
        for field in DROP_CSV_FIELDS:
            chinese_name, calculation, meaning, formula = DROP_FIELD_DEFINITIONS[field]
            writer.writerow({
                "parameter": field,
                "chinese_name": chinese_name,
                "calculation_description": calculation,
                "meaning": meaning,
                "formula": formula,
            })


def _ok_rows(rows):
    return [row for row in rows if row.get("status") == "ok"]


def _float(row, field):
    value = row.get(field, "")
    if value == "":
        return 0.0
    return float(value)


def _mean(rows, field):
    if not rows:
        return 0.0
    return sum(_float(row, field) for row in rows) / len(rows)


def _best_long_chain_row(rows):
    return max(
        rows,
        key=lambda row: (
            _float(row, "effective_chain_coverage_ge_16_gain"),
            _float(row, "effective_chain_coverage_ge_8_gain"),
            _float(row, "effective_chain_coverage_ge_32_gain"),
            _float(row, "effective_long_chain_score_gain_per_drop_percent"),
            _float(row, "frontier_safe_boundary_cut_reduction"),
        ),
    )


def _long_chain_verdict(row):
    ge8_gain = _float(row, "effective_chain_coverage_ge_8_gain")
    ge16_gain = _float(row, "effective_chain_coverage_ge_16_gain")
    ge32_gain = _float(row, "effective_chain_coverage_ge_32_gain")
    long_score = _float(row, "effective_long_chain_score_gain_per_drop_percent")

    if ge16_gain >= 0.005 or ge8_gain >= 0.01 or ge32_gain >= 0.002:
        return "effective"
    if ge16_gain < -1e-9 or ge8_gain < -1e-9 or long_score < -1e-9:
        return "regressed"
    return "weak-or-ineffective"


def write_summary(summary_path: Path, rows):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    ok_rows = _ok_rows(rows)
    failed_rows = [row for row in rows if row.get("status") == "failed"]
    methods = sorted({row["drop_method"] for row in ok_rows})
    matrices = sorted({row["matrix_name"] for row in ok_rows})

    lines = [
        "Long-chain-oriented drop experiment summary",
        "",
        f"total_rows: {len(rows)}",
        f"ok_rows: {len(ok_rows)}",
        f"failed_rows: {len(failed_rows)}",
        f"completed_matrices: {len(matrices)}",
        "",
        "Method averages focused on effective long chains:",
    ]

    method_avg = {}
    for method in methods:
        method_rows = [row for row in ok_rows if row["drop_method"] == method]
        if not method_rows:
            continue
        avg_ge8_gain = _mean(method_rows, "effective_chain_coverage_ge_8_gain")
        avg_ge16_gain = _mean(method_rows, "effective_chain_coverage_ge_16_gain")
        avg_ge32_gain = _mean(method_rows, "effective_chain_coverage_ge_32_gain")
        avg_ge8_per_percent = _mean(
            method_rows,
            "effective_chain_coverage_ge_8_gain_per_drop_percent",
        )
        avg_long_score_per_percent = _mean(
            method_rows,
            "effective_long_chain_score_gain_per_drop_percent",
        )
        avg_count_ge16_gain = _mean(method_rows, "effective_chain_count_ge_16_gain")
        avg_cut_reduction = _mean(method_rows, "frontier_safe_boundary_cut_reduction")
        method_avg[method] = {
            "ge8_gain": avg_ge8_gain,
            "ge16_gain": avg_ge16_gain,
            "ge32_gain": avg_ge32_gain,
            "ge8_per_percent": avg_ge8_per_percent,
            "long_score_per_percent": avg_long_score_per_percent,
        }
        lines.append(
            f"- {method}: avg_ge8_coverage_gain={avg_ge8_gain:.6g}, "
            f"avg_ge16_coverage_gain={avg_ge16_gain:.6g}, "
            f"avg_ge32_coverage_gain={avg_ge32_gain:.6g}, "
            f"avg_ge8_gain_per_drop_percent={avg_ge8_per_percent:.6g}, "
            f"avg_long_score_gain_per_drop_percent={avg_long_score_per_percent:.6g}, "
            f"avg_count_ge16_gain={avg_count_ge16_gain:.6g}, "
            f"avg_boundary_cut_reduction={avg_cut_reduction:.6g}"
        )

    non_baseline_avg = {
        method: values
        for method, values in method_avg.items()
        if method != "no-drop"
    }
    if non_baseline_avg:
        best_avg_method = max(
            non_baseline_avg,
            key=lambda method: (
                non_baseline_avg[method]["ge16_gain"],
                non_baseline_avg[method]["ge8_gain"],
                non_baseline_avg[method]["long_score_per_percent"],
            ),
        )
        lines.extend([
            "",
            "Best average method by effective long-chain gain:",
            (
                f"- {best_avg_method}: "
                f"avg_ge8_gain={non_baseline_avg[best_avg_method]['ge8_gain']:.6g}, "
                f"avg_ge16_gain={non_baseline_avg[best_avg_method]['ge16_gain']:.6g}, "
                f"avg_long_score_per_drop_percent="
                f"{non_baseline_avg[best_avg_method]['long_score_per_percent']:.6g}"
            ),
        ])

    lines.extend(["", "Direct method comparison:"])
    long_avg = method_avg.get("long-chain-aware-drop")
    if long_avg is None:
        lines.append("- long-chain-aware-drop did not produce completed rows.")
    else:
        for baseline in ("scan-chain-aware", "magnitude-drop", "wavefront-oriented"):
            baseline_avg = method_avg.get(baseline)
            if baseline_avg is None:
                lines.append(f"- vs {baseline}: baseline has no completed rows.")
                continue
            lines.append(
                f"- vs {baseline}: "
                f"delta_avg_ge8_gain={long_avg['ge8_gain'] - baseline_avg['ge8_gain']:.6g}, "
                f"delta_avg_ge16_gain={long_avg['ge16_gain'] - baseline_avg['ge16_gain']:.6g}, "
                f"delta_long_score_per_percent="
                f"{long_avg['long_score_per_percent'] - baseline_avg['long_score_per_percent']:.6g}"
            )

    param_groups = {}
    for row in ok_rows:
        if row.get("drop_method") == "no-drop":
            continue
        key = (
            row["drop_method"],
            row["drop_ratio_target"],
            row["risk_threshold"],
            row["max_drop_per_row"],
        )
        param_groups.setdefault(key, []).append(row)
    top_params = sorted(
        param_groups.items(),
        key=lambda item: (
            _mean(item[1], "effective_chain_coverage_ge_16_gain"),
            _mean(item[1], "effective_chain_coverage_ge_8_gain"),
            _mean(item[1], "effective_long_chain_score_gain_per_drop_percent"),
        ),
        reverse=True,
    )[:5]
    if top_params:
        lines.extend(["", "Most stable parameter settings in this batch:"])
        for (method, ratio, risk, max_drop), param_rows in top_params:
            lines.append(
                f"- method={method}, ratio={ratio}, risk={risk}, "
                f"max_drop_per_row={max_drop}: "
                f"avg_ge8_gain={_mean(param_rows, 'effective_chain_coverage_ge_8_gain'):.6g}, "
                f"avg_ge16_gain={_mean(param_rows, 'effective_chain_coverage_ge_16_gain'):.6g}, "
                f"avg_long_score_per_drop_percent="
                f"{_mean(param_rows, 'effective_long_chain_score_gain_per_drop_percent'):.6g}"
            )

    lines.extend(["", "Per-matrix best rows:"])
    verdict_counts = {
        "effective": 0,
        "weak-or-ineffective": 0,
        "regressed": 0,
    }
    support_matrices = []
    failing_matrices = []
    for matrix in matrices:
        matrix_rows = [
            row for row in ok_rows
            if row["matrix_name"] == matrix and row["drop_method"] != "no-drop"
        ]
        if not matrix_rows:
            continue
        best = _best_long_chain_row(matrix_rows)
        verdict = _long_chain_verdict(best)
        verdict_counts[verdict] += 1
        if verdict == "effective":
            support_matrices.append(matrix)
        else:
            failing_matrices.append(matrix)
        lines.append(
            f"- {matrix}: {verdict}; best={best['drop_method']}, "
            f"ratio={best['drop_ratio_target']}, risk={best['risk_threshold']}, "
            f"ge8_gain={_float(best, 'effective_chain_coverage_ge_8_gain'):.6g}, "
            f"ge16_gain={_float(best, 'effective_chain_coverage_ge_16_gain'):.6g}, "
            f"ge32_gain={_float(best, 'effective_chain_coverage_ge_32_gain'):.6g}, "
            f"ge8_after={_float(best, 'effective_chain_coverage_ge_8_after'):.6g}, "
            f"ge16_after={_float(best, 'effective_chain_coverage_ge_16_after'):.6g}, "
            f"long_score_gain_per_drop_percent="
            f"{_float(best, 'effective_long_chain_score_gain_per_drop_percent'):.6g}"
        )

    lines.extend([
        "",
        "Matrix verdict counts:",
        f"- effective: {verdict_counts['effective']}",
        f"- weak-or-ineffective: {verdict_counts['weak-or-ineffective']}",
        f"- regressed: {verdict_counts['regressed']}",
    ])

    lines.extend(["", "Final decision:"])
    enough_support = (
        verdict_counts["effective"] >= 2
        and verdict_counts["effective"] >= max(1, len(matrices) // 2)
    )
    if enough_support:
        lines.append(
            "- Current static evidence supports considering a scan-chain executor "
            "for the effective matrices, but only after numerical-quality checks."
        )
    else:
        lines.append(
            "- Current static evidence is weak-or-ineffective for executor work: "
            "do not start a GPU scan-chain executor yet."
        )

    if long_avg is not None:
        scan_avg = method_avg.get("scan-chain-aware")
        if scan_avg is None:
            lines.append("- long-chain-aware-drop could not be compared with scan-chain-aware.")
        elif (
            long_avg["ge8_gain"] > scan_avg["ge8_gain"]
            or long_avg["ge16_gain"] > scan_avg["ge16_gain"]
        ):
            lines.append(
                "- long-chain-aware-drop improves at least one long-chain gain "
                "metric over scan-chain-aware."
            )
        else:
            lines.append(
                "- long-chain-aware-drop is not clearly better than scan-chain-aware "
                "on the completed batch."
            )

        magnitude_avg = method_avg.get("magnitude-drop")
        if magnitude_avg is not None:
            if (
                long_avg["ge8_gain"] > magnitude_avg["ge8_gain"]
                or long_avg["ge16_gain"] > magnitude_avg["ge16_gain"]
            ):
                lines.append(
                    "- long-chain-aware-drop shows structural long-chain gain over magnitude-drop."
                )
            else:
                lines.append(
                    "- long-chain-aware-drop does not beat magnitude-drop on effective >=8/>=16 coverage gain."
                )

        wavefront_avg = method_avg.get("wavefront-oriented")
        if wavefront_avg is not None:
            if (
                long_avg["ge8_gain"] > wavefront_avg["ge8_gain"]
                or long_avg["ge16_gain"] > wavefront_avg["ge16_gain"]
            ):
                lines.append(
                    "- long-chain-aware-drop is more scan-chain-targeted than wavefront-oriented in at least one long-chain gain metric."
                )
            else:
                lines.append(
                    "- long-chain-aware-drop is not clearly better than wavefront-oriented on effective long-chain coverage."
                )

    lines.append(
        "- The primary decision metric is effective_chain_coverage_ge_8/16_gain; "
        "strict total coverage gains are secondary and are not sufficient evidence."
    )

    if support_matrices:
        lines.append(f"- Matrices supporting executor follow-up: {', '.join(support_matrices)}")
    else:
        lines.append("- Matrices supporting executor follow-up: none")
    if failing_matrices:
        lines.append(f"- Matrices currently not supporting the direction: {', '.join(failing_matrices)}")

    if failed_rows:
        lines.extend(["", "Failures:"])
        for row in failed_rows:
            lines.append(f"- {row.get('matrix_name')}: {row.get('error')}")

    lines.extend([
        "",
        "Decision rule used here:",
        "- effective if ge16 coverage gain >= 0.005, or ge8 coverage gain >= 0.01, or ge32 coverage gain >= 0.002.",
        "- regressed if the best non-baseline row has negative long-chain coverage/score gain.",
        "- otherwise weak-or-ineffective.",
    ])

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_by_method_ratio(rows, value_field):
    grouped = {}
    for row in _ok_rows(rows):
        if row.get("drop_method") == "no-drop":
            continue
        key = (row["drop_method"], float(row["drop_ratio_target"]))
        grouped.setdefault(key, []).append(_float(row, value_field))
    return {
        key: sum(values) / len(values)
        for key, values in grouped.items()
        if values
    }


def _aggregate_by_method(rows, value_field):
    grouped = {}
    for row in _ok_rows(rows):
        key = row["drop_method"]
        grouped.setdefault(key, []).append(_float(row, value_field))
    return {
        key: sum(values) / len(values)
        for key, values in grouped.items()
        if values
    }


def write_plots(fig_dir: Path, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed; skip plots")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("effective_chain_coverage_ge_8", "drop_ratio_vs_effective_ge8.png", "Effective chain coverage >= 8"),
        ("effective_chain_coverage_ge_16", "drop_ratio_vs_effective_ge16.png", "Effective chain coverage >= 16"),
        ("effective_chain_coverage_ge_8_gain_per_drop_percent", "drop_ratio_vs_ge8_gain_per_drop_percent.png", "Coverage >= 8 gain per dropped 1%"),
        ("effective_chain_coverage_ge_16_gain_per_drop_percent", "drop_ratio_vs_ge16_gain_per_drop_percent.png", "Coverage >= 16 gain per dropped 1%"),
        ("effective_long_chain_score_gain_per_drop_percent", "drop_ratio_vs_long_score_gain_per_drop_percent.png", "Weighted long-chain score gain per dropped 1%"),
        ("strict_chain_len_ge_16_after", "drop_ratio_vs_chain_count_ge16.png", "Chain count >= 16 after drop"),
    ]

    for field, filename, ylabel in plot_specs:
        values = _aggregate_by_method_ratio(rows, field)
        if not values:
            continue
        methods = sorted({method for method, _ in values})
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method in methods:
            points = sorted(
                (ratio, value)
                for (item_method, ratio), value in values.items()
                if item_method == method
            )
            ax.plot(
                [ratio for ratio, _ in points],
                [value for _, value in points],
                marker="o",
                label=method,
            )
        ax.set_xlabel("drop ratio target")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=180)
        plt.close(fig)

    thresholds = (8, 16, 32)
    distribution = {
        threshold: _aggregate_by_method(rows, f"strict_chain_len_ge_{threshold}_after")
        for threshold in thresholds
    }
    methods = sorted({
        method
        for values in distribution.values()
        for method in values
    })
    if methods:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        x = np.arange(len(methods))
        width = 0.24
        for offset_index, threshold in enumerate(thresholds):
            values = [
                distribution[threshold].get(method, 0.0)
                for method in methods
            ]
            ax.bar(
                x + (offset_index - 1) * width,
                values,
                width=width,
                label=f"length >= {threshold}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel("average chain count after drop")
        ax.set_title("Strict-chain length distribution by method")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "chain_length_distribution_by_method.png", dpi=180)
        plt.close(fig)

    best_rows = []
    for matrix in sorted({row["matrix_name"] for row in _ok_rows(rows)}):
        matrix_rows = [
            row for row in _ok_rows(rows)
            if row["matrix_name"] == matrix and row.get("drop_method") != "no-drop"
        ]
        if matrix_rows:
            best_rows.append(_best_long_chain_row(matrix_rows))
    if best_rows:
        labels = [row["matrix_name"] for row in best_rows]
        ge8_values = [_float(row, "effective_chain_coverage_ge_8_gain") for row in best_rows]
        ge16_values = [_float(row, "effective_chain_coverage_ge_16_gain") for row in best_rows]
        methods = [row["drop_method"] for row in best_rows]

        fig, ax = plt.subplots(figsize=(max(7.5, 0.5 * len(labels)), 4.8))
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, ge8_values, width=width, label="best >=8 gain")
        ax.bar(x + width / 2, ge16_values, width=width, label="best >=16 gain")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{label}\n{method}" for label, method in zip(labels, methods, strict=False)],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel("coverage gain")
        ax.set_title("Best long-chain gain per matrix")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "best_method_by_matrix_long_chain_gain.png", dpi=180)
        plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run drop-element scan-chain static experiments."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help=(
            "Input .mtx, matrix/factor directory, or bare matrix name. "
            "Defaults to data/factors/spcg for --source factor-l, otherwise "
            "data/matrices/real."
        ),
    )
    parser.add_argument(
        "--source",
        default="factor-l",
        choices=["factor-l", "spcg-ilu-l", "matrix-lower"],
        help="Experiment data source; factor-l is the default for preconditioner-focused tests.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DROP_METHODS),
        help=f"Comma-separated drop methods. Available: {','.join(DROP_METHODS)}",
    )
    parser.add_argument(
        "--drop-ratios",
        default=",".join(f"{ratio:g}" for ratio in DEFAULT_DROP_RATIOS),
        help="Comma-separated target drop ratios, default: 0.01,0.02,0.05,0.10",
    )
    parser.add_argument(
        "--risk-thresholds",
        default=",".join(f"{value:g}" for value in DEFAULT_RISK_THRESHOLDS),
        help="Comma-separated relative risk thresholds, default: 1e-4,1e-3,1e-2",
    )
    parser.add_argument(
        "--max-drop-per-row",
        default=",".join(str(value) for value in DEFAULT_MAX_DROP_PER_ROW),
        help="Comma-separated per-row drop limits. Use -1 for no limit.",
    )
    parser.add_argument(
        "--allow-drop-row-max-predecessor",
        action="store_true",
        help="Allow deleting the largest-magnitude predecessor in a row.",
    )
    parser.add_argument(
        "--allow-empty-rows",
        action="store_true",
        help="Allow deleting all strict-lower predecessors of a row.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_DROP_OUT),
        help=f"Output CSV path (default: {DEFAULT_DROP_OUT})",
    )
    parser.add_argument(
        "--replace-out",
        action="store_true",
        help=(
            "Rewrite --out from only this run. By default, existing rows are "
            "kept and rows with the same matrix/source/path/method/parameters "
            "are updated."
        ),
    )
    parser.add_argument(
        "--summary-out",
        default=str(DEFAULT_DROP_SUMMARY_OUT),
        help=f"Summary text path (default: {DEFAULT_DROP_SUMMARY_OUT})",
    )
    parser.add_argument(
        "--fig-dir",
        default=str(DEFAULT_DROP_FIG_DIR),
        help=f"Figure output directory (default: {DEFAULT_DROP_FIG_DIR})",
    )
    parser.add_argument(
        "--definitions-out",
        default=str(DEFAULT_DROP_DEFINITIONS_OUT),
        help=(
            "Output CSV path for drop experiment field definitions "
            f"(default: {DEFAULT_DROP_DEFINITIONS_OUT})"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of input matrices/factors to process.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plot generation.",
    )
    parser.add_argument(
        "--factor-dir",
        default=str(base.DEFAULT_FACTOR_DIR),
        help=f"Directory for generated SPCG-style factors (default: {base.DEFAULT_FACTOR_DIR})",
    )
    parser.add_argument(
        "--spcg-method",
        default="spilu",
        choices=["spilu", "splu"],
        help="SPCG-style factorization method for --source spcg-ilu-l",
    )
    parser.add_argument(
        "--fill-factor",
        type=float,
        default=10.0,
        help="SuperLU fill_factor for --spcg-method spilu",
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
        help="Optional value-threshold sparsification before SPCG factorization.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.spcg_sparsify_percentage < 0 or args.spcg_sparsify_percentage >= 1:
        raise ValueError("--spcg-sparsify-percentage must be in [0, 1)")
    output_path, summary_path, fig_dir, definitions_path, rows = run_experiment(args)
    print(f"Done. Wrote {len(rows)} rows: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"Field definitions: {definitions_path}")
    if not args.no_plots:
        print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()

# 删元素制造 Scan 链静态实验框架

本文档说明 `scan_SpTRSV` 新增的 drop-element scan-chain 实验框架。目标不是实现 GPU 求解器，而是先判断一个前置问题：

```text
对真实 L 因子删除少量低风险元素，能否显著制造更多有执行价值的 scan 链？
```

当前阶段不看 strict 总链覆盖率是否上升，而看长度 `>=8/>=16` 的有效长链覆盖率是否稳定上升。如果有效长链覆盖率没有明显提升，后续 scan-chain executor 没有足够依据继续投入。

## 新增内容

- `src/scan_sptrsv/drop_experiments.py`：核心实验模块，负责读取带数值的严格下三角依赖图、选择删除边、重算 strict/frontier-safe 链并输出统计。
- `scripts/run_drop_experiment.py`：命令行入口。
- `results/drop_experiments/`：默认总 CSV 和 summary 输出目录。
- `results/definitions/drop_experiment_fields.csv`：运行时自动生成的字段定义表。
- `figures/drop_experiments/`：默认图表输出目录。

## 实验对象

优先使用真实或生成的 `L` 因子：

- `factor-l`：直接分析已有 `L.mtx`，默认输入目录是 `data/factors/spcg/`。
- `spcg-ilu-l`：从 `data/matrices/real/` 的原矩阵生成 SPCG 风格 SuperLU `L/U`，再分析 `L`。
- `matrix-lower`：直接抽原矩阵严格下三角，只作为辅助对照。

## 删除策略

`no-drop`

不删除任何 strict-lower 元素，作为原始基线。所有 before/after 指标相同。

`magnitude-drop`

普通删小值基线。每条边的风险定义为：

```text
risk(i,j) = |L(i,j)| / max_k |L(i,k)|
```

只允许删除 `risk <= risk_threshold` 的边，并优先删除 `risk` 更小的边。

`wavefront-oriented`

波前导向基线。先按 frontier-safe 的唯一最深前驱逻辑计算依赖层级 `level`，再倾向删除更深层前驱或候选 parent 边。它的目的不是直接制造 strict 单前驱链，而是模拟“减少依赖层数或波前阻塞”的 SPCG 类思路。

`scan-chain-aware`

长链偏置的 scan-chain-aware 基线。它不再只优化 strict 总覆盖，也不只追求把多前驱行变成单前驱行；评分会估计某行删到单前驱后能接出的局部潜在链长，并优先选择更可能形成 `>=8/>=16/>=32` 长链的候选。

- 优先处理删除预算足以转成单前驱的行。
- 保留能形成最长局部链的主前驱候选，默认不删该行最大幅值前驱。
- 更偏向删除非 frontier 主依赖的边，减少 frontier-safe 的切链来源。
- 长度小于 8 的局部收益会被降权。
- 遵守风险阈值、每行最大删除数、禁止删空前驱等安全约束。

`long-chain-aware-drop`

专门制造有效长链的新策略。它比 `scan-chain-aware` 更激进地围绕有效长链排序：

- 对每行估计 `upstream_len(kept_pred) + downstream_len(row)` 的局部潜在链长。
- 只高优先级处理潜在链长 `>=8` 的候选行。
- 对潜在长度 `>=16`、`>=32` 给予更高权重。
- 如果一行需要删除多条边才能变成单前驱，则按所需删除数摊薄收益。
- 若 `max_drop_per_row=1` 下无法把高前驱数行转成单前驱，该策略可能实际删除为 0，这是有效负结果。

## 默认参数

默认命令会扫描：

```text
drop ratios:       0.01, 0.02, 0.05, 0.10
risk thresholds:   1e-4, 1e-3, 1e-2
max drop per row:  1
keep row max pred: yes
forbid empty rows: yes
methods:           no-drop, magnitude-drop, wavefront-oriented, scan-chain-aware, long-chain-aware-drop
source:            factor-l
```

`drop_ratio_actual` 可能小于 `drop_ratio_target`，因为低风险候选不足或安全约束阻止继续删除。

## 单矩阵运行

已有 `L` 因子：

```bash
python3 scripts/run_drop_experiment.py \
  data/factors/spcg/tmt_sym/spcg_spilu_l_tmt_sym_fill10_dropdefault_sp0.mtx \
  --source factor-l
```

原矩阵直接生成 SPCG 风格 `L` 后分析：

```bash
python3 scripts/run_drop_experiment.py tmt_sym \
  --source spcg-ilu-l \
  --spcg-method spilu \
  --fill-factor 10
```

小矩阵 smoke test：

```bash
python3 scripts/run_drop_experiment.py data/matrices/regression \
  --source matrix-lower \
  --drop-ratios 0.1 \
  --risk-thresholds 1 \
  --methods no-drop,magnitude-drop,wavefront-oriented,scan-chain-aware,long-chain-aware-drop \
  --no-plots \
  --out results/drop_experiments/regression_long_chain_smoke.csv \
  --summary-out results/drop_experiments/regression_long_chain_smoke_summary.txt
```

## 批量运行

分析一批已有 L 因子：

```bash
python3 scripts/run_drop_experiment.py data/factors/spcg \
  --source factor-l
```

当前长链导向批量实验示例：

```bash
python3 scripts/run_drop_experiment.py data/factors/spcg \
  --source factor-l \
  --methods no-drop,magnitude-drop,wavefront-oriented,scan-chain-aware,long-chain-aware-drop \
  --drop-ratios 0.01,0.05 \
  --risk-thresholds 0.1 \
  --max-drop-per-row 1,-1 \
  --out results/drop_experiments/long_chain_factor_batch.csv \
  --summary-out results/drop_experiments/long_chain_factor_batch_summary.txt \
  --fig-dir figures/drop_experiments/long_chain_factor_batch
```

从真实矩阵批量生成并分析 SPCG `L`：

```bash
python3 scripts/run_drop_experiment.py data/matrices/real \
  --source spcg-ilu-l \
  --spcg-method spilu \
  --fill-factor 10
```

批处理遇到失败矩阵不会中断全局实验，会在总 CSV 中写入 `status=failed` 和 `error`。

## 输出文件

默认输出：

```text
results/drop_experiments/drop_scan_chain_sweep.csv
results/drop_experiments/drop_scan_chain_summary.txt
results/definitions/drop_experiment_fields.csv
figures/drop_experiments/drop_ratio_vs_effective_ge8.png
figures/drop_experiments/drop_ratio_vs_effective_ge16.png
figures/drop_experiments/drop_ratio_vs_ge8_gain_per_drop_percent.png
figures/drop_experiments/drop_ratio_vs_ge16_gain_per_drop_percent.png
figures/drop_experiments/drop_ratio_vs_long_score_gain_per_drop_percent.png
figures/drop_experiments/drop_ratio_vs_chain_count_ge16.png
figures/drop_experiments/chain_length_distribution_by_method.png
figures/drop_experiments/best_method_by_matrix_long_chain_gain.png
```

可以用 `--out`、`--summary-out`、`--definitions-out`、`--fig-dir` 改输出位置。

默认写 CSV 使用 upsert 规则：如果旧表里已经有相同
`matrix_name/source/input_path/drop_method/drop_ratio_target/risk_threshold/max_drop_per_row/keep_row_max_predecessor/forbid_empty_rows`
组合，则更新这一行；其他旧行保留。需要只保留本次运行结果时加：

```bash
--replace-out
```

## 关键字段

- `drop_method`：删除策略。
- `drop_ratio_target` / `drop_ratio_actual`：目标和实际删除比例。
- `dropped_nnz` / `dropped_rows`：删除边数和涉及行数。
- `strict_lower_nnz_before` / `strict_lower_nnz_after`：删除前后 strict-lower 边数。
- `strict_chain_row_coverage_before` / `strict_chain_row_coverage_after`：删除前后 strict 链覆盖行数。
- `frontier_safe_chain_row_coverage_before` / `frontier_safe_chain_row_coverage_after`：删除前后 frontier-safe 链覆盖行数。
- `effective_chain_rows_ge_8/16/32`：删除后长度至少 8/16/32 的 strict 链覆盖行数。
- `effective_chain_coverage_ge_8/16/32`：对应覆盖率。
- `effective_chain_coverage_ge_8/16/32_gain`：删除前后有效长链覆盖率增量，当前最重要判断字段。
- `effective_chain_count_ge_8/16/32_gain`：删除前后有效长链数量增量。
- `avg_effective_chain_length_ge_8/16/32_after`：删除后有效长链平均长度。
- `effective_long_chain_score_gain_per_drop_percent`：每删除 1% strict-lower 非零元带来的加权长链收益。
- `multi_pred_to_single_pred_rows`：多前驱行被删成单前驱行的数量。
- `frontier_safe_boundary_cut_reduction`：frontier-safe 切链点减少量。
- `chain_gain_per_dropped_edge`：每删一条 strict-lower 边换来的 strict 覆盖行数收益。
- `strict_chain_coverage_gain_per_drop_ratio`：每 1.0 删除比例换来的 strict 覆盖率收益；看每删除 1% 时可除以 100 理解。
- `strict_chain_coverage_gain_per_drop_percent`：每删除 1% strict-lower 非零元换来的 strict 覆盖率收益。
- `effective_chain_coverage_ge_8/16/32_gain_per_drop_percent`：每删除 1% strict-lower 非零元换来的长度至少 8/16/32 有效链覆盖率收益。

完整字段定义由程序输出到：

```text
results/definitions/drop_experiment_fields.csv
```

## 当前判断规则

summary 里目前使用长链导向规则：

```text
effective if:
  effective_chain_coverage_ge_16_gain >= 0.005
  or effective_chain_coverage_ge_8_gain >= 0.01
  or effective_chain_coverage_ge_32_gain >= 0.002
```

满足该条件的矩阵标为 `effective`，负收益标为 `regressed`，其余标为 `weak-or-ineffective`。这个阈值只是筛查规则，不是论文结论。真正判断应同时看：

- scan-chain-aware 是否在同等 `drop_ratio_actual` 下优于 magnitude-drop 和 wavefront-oriented。
- long-chain-aware-drop 是否明显优于原 scan-chain-aware。
- `effective_chain_coverage_ge_8/16/32_gain` 是否明显上升。
- `effective_chain_rows_ge_8/16/32_gain_per_dropped_edge` 是否为正且足够大。
- `frontier_safe_boundary_cut_reduction` 是否减少，而不是只把短链数量刷高。

如果大多数真实 `L` 因子在 1% 到 10% 删除比例内仍没有有效长链覆盖率提升，应明确写成负结果：当前静态证据不支持继续实现 scan-chain executor。

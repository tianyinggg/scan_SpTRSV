# scan_SpTRSV

`scan_SpTRSV` analyzes strict single-predecessor scan chains for sparse triangular dependency structures. It does not reimplement the `2020-lu-sptrsv` level-set solver; it extracts dependency statistics from Matrix Market inputs with a streaming reader.

`scan_SpTRSV` 用于分析稀疏三角依赖结构中的严格单前驱 scan 链。它不是 `2020-lu-sptrsv` 的 level-set 求解器复刻版，而是通过流式 Matrix Market 读取器抽取依赖统计参数。

## Layout

## 目录结构

```text
scan_SpTRSV/
├── README.md
├── src/
├── scripts/
├── results/
├── figures/
├── docs/
├── experiments/
├── dump/
├── datasets/
└── .gitignore
```

- `src/scan_sptrsv/`: core implementation
- `src/scan_sptrsv/`: 核心实现
- `scripts/`: runnable entry points
- `scripts/`: 可直接运行的脚本入口
- `results/`: stable result tables
- `results/`: 稳定保存的结果表
- `figures/`: exported figures for reports or papers
- `figures/`: 用于论文或报告的图表
- `docs/`: notes, metric definitions, drafts
- `docs/`: 说明文档、指标定义、草稿和笔记
- `experiments/`: temporary or small-scale experiment artifacts
- `experiments/`: 临时实验或小规模回归结果
- `dump/`: intermediate dumps and debug data
- `dump/`: 中间 dump 和调试数据
- `datasets/`: Matrix Market inputs
- `datasets/`: Matrix Market 输入数据

## Main Entry Point

## 主入口

Run the analyzer through:

运行方式：

```bash
python3 scripts/analyze_scan_chains.py mawi_201512020030
```

The default source is `matrix-lower`: the analyzer streams the original Matrix
Market input and extracts strict lower-triangular dependencies directly.

默认数据源是 `matrix-lower`：程序流式读取原始 Matrix Market 矩阵，并直接抽取严格下三角依赖。

A bare matrix name resolves to:

裸矩阵名会解析到：

```text
datasets/<name>/<name>.mtx
```

Default output:

默认输出：

```text
results/bigtest_end.csv
```

The output file is updated by `matrix_name`: new matrices are appended; reruns replace the existing row for the same matrix.

结果文件按 `matrix_name` 做更新：新矩阵追加新行，重跑同名矩阵会替换对应结果行。

For small regression tests, run:

小矩阵回归测试运行方式：

```bash
python3 scripts/analyze_scan_chains.py --small-test
```

Small-test output:

小矩阵测试输出：

```text
experiments/scan_chain_stats_t01_t05_test.csv
```

This file uses the same `matrix_name` update rule as `results/bigtest_end.csv`.

该文件使用和 `results/bigtest_end.csv` 相同的按 `matrix_name` 更新规则。

## SPCG-Style Factor Inputs

## SPCG 风格真实三角因子输入

`scan_SpTRSV` also supports a parallel data path based on SPCG-style real
triangular factors. This path does not replace `matrix-lower`; it generates or
reads an actual `L` factor and then feeds that `L` into the same strict-chain
statistics.

`scan_SpTRSV` 也支持一条并行的 SPCG 风格真实三角因子路径。它不会替代 `matrix-lower`；这条路径会生成或读取真实 `L` 因子，再把该 `L` 送入同一套严格链统计逻辑。

Generate SuperLU `L/U` factors with SPCG-like settings and analyze `L`:

使用接近 SPCG 的 SuperLU 设置生成 `L/U` 因子并分析 `L`：

```bash
python3 scripts/analyze_scan_chains.py kkt_power --source spcg-ilu-l
```

Default generated factors:

默认生成的因子位置：

```text
dump/spcg_factors/
```

Default output for this source:

这条路径的默认输出：

```text
results/spcg_factor_chain_stats.csv
```

Useful options:

常用选项：

```bash
python3 scripts/analyze_scan_chains.py kkt_power \
  --source spcg-ilu-l \
  --spcg-method spilu \
  --fill-factor 20 \
  --drop-tol 1e-12 \
  --spcg-sparsify-percentage 0.05
```

Analyze an existing lower factor `.mtx` directly:

直接分析已有下三角因子 `.mtx`：

```bash
python3 scripts/analyze_scan_chains.py dump/spcg_factors/spcg_spilu_l_kkt_power_fill10_dropdefault_sp0.mtx \
  --source factor-l
```

## Current Outputs

## 当前输出

- `results/bigtest_end.csv`: aggregated large-matrix statistics
- `results/bigtest_end.csv`: 汇总后的大矩阵统计结果
- `results/bigtest_end_parameter_definitions.csv`: field source, meaning, calculation description, and formula
- `results/bigtest_end_parameter_definitions.csv`: 每个字段的来源、意义、计算描述与公式

## Notes

## 说明

- Large `.mtx` files live under `datasets/`
- 大矩阵 `.mtx` 文件放在 `datasets/`
- Dump files should go under `dump/`
- 中间 dump 文件放在 `dump/`
- Small regression artifacts should go under `experiments/`
- 小规模回归结果放在 `experiments/`

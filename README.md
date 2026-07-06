# scan_SpTRSV

`scan_SpTRSV` analyzes scan opportunities in sparse triangular dependency
structures. The project separates data sources from analysis methods.

`scan_SpTRSV` 用于分析稀疏三角依赖结构中适合做 scan 的链。当前工程已经把“数据来源”和“统计方法”彻底分开：

- Data source: original matrices or generated SPCG-style `L/U` factors
- 数据来源：原始矩阵，或由 SPCG 风格因子化生成的真实 `L/U` 因子
- Analysis method: strict single-predecessor chains or frontier-safe chains
- 统计方法：严格单前驱链，或 frontier-safe 链

## Directory Layout

## 目录结构

```text
scan_SpTRSV/
├── data/
│   ├── matrices/
│   │   ├── regression/      # small correctness matrices
│   │   └── real/            # large real Matrix Market inputs
│   └── factors/
│       └── spcg/            # generated SPCG-style L/U factors
├── results/
│   ├── strict/              # strict-chain result tables
│   ├── frontier_safe/       # strict + frontier-safe result tables
│   ├── regression/          # small regression outputs
│   └── definitions/         # field definitions
├── src/scan_sptrsv/         # core implementation
├── scripts/                 # runnable entry points
├── docs/                    # notes and method docs
├── figures/                 # plots and exported figures
└── legacy/                  # old outputs and old bundles, not active paths
```

中文说明：

- `data/matrices/regression/`：小矩阵测试集，用于验证统计逻辑是否正确。
- `data/matrices/real/`：真实大矩阵输入，例如 `tmt_sym`、`mawi_201512020030`。
- `data/factors/spcg/`：由 SPCG 风格 `spilu/splu` 生成的真实三角因子，按矩阵名分目录保存。
- `results/strict/`：严格单前驱链统计结果。
- `results/frontier_safe/`：frontier-safe 统计结果，会同时包含 strict 字段和 frontier-safe 字段。
- `results/regression/`：小测试集输出。
- `results/definitions/`：字段定义和说明。
- `legacy/`：旧结果和旧 bundle，仅作历史保留，不再作为默认运行路径。

## Data Sources

## 数据来源

### `matrix-lower`

`matrix-lower` reads an original matrix and extracts strict lower-triangular
dependencies directly.

`matrix-lower` 直接读取原始矩阵，并从中抽取严格下三角依赖。这是“原矩阵直接拆下三角”的路径。

```bash
python3 scripts/analyze_scan_chains.py tmt_sym --source matrix-lower
```

Bare matrix names resolve to:

裸矩阵名会解析到：

```text
data/matrices/real/<name>/<name>.mtx
```

Default output:

默认输出：

```text
results/strict/matrix_lower.csv
```

### `spcg-ilu-l`

`spcg-ilu-l` generates SPCG-style SuperLU `L/U` factors, stores them under
`data/factors/spcg/<matrix>/`, and analyzes the generated `L`.

`spcg-ilu-l` 会先对原矩阵做 SPCG 风格的 SuperLU 因子化，生成真实 `L/U`，保存到 `data/factors/spcg/<matrix>/`，然后只分析生成出来的真实 `L` 因子。

```bash
python3 scripts/analyze_scan_chains.py tmt_sym --source spcg-ilu-l
```

Default output:

默认输出：

```text
results/strict/spcg_factor_l.csv
```

Useful options:

常用参数：

```bash
python3 scripts/analyze_scan_chains.py tmt_sym \
  --source spcg-ilu-l \
  --spcg-method spilu \
  --fill-factor 20 \
  --drop-tol 1e-12 \
  --spcg-sparsify-percentage 0.05
```

### `factor-l`

`factor-l` analyzes an existing lower factor `.mtx` directly.

`factor-l` 直接分析已有的下三角 `L.mtx` 文件。如果传入的是一个因子目录，程序会只筛选 `L` 因子，避免把同目录里的 `U` 当成下三角矩阵分析。

```bash
python3 scripts/analyze_scan_chains.py \
  data/factors/spcg/tmt_sym/spcg_spilu_l_tmt_sym_fill10_dropdefault_sp0.mtx \
  --source factor-l
```

## Analysis Methods

## 统计方法

### Strict Chains

Strict chains use the conservative rule:

严格链使用最保守的规则：

```text
pred_count[row] == 1
```

That means a row can be part of a strict scan chain only when it has exactly one
lower-triangular predecessor.

也就是说，某一行只有在严格下三角中恰好只有一个前驱时，才可以进入 strict scan 链。这种口径解释简单，但通常覆盖率偏低。

Strict mode is the default method and writes only strict-chain fields.

strict 模式是默认统计方法，只输出 strict 相关字段。

### Frontier-Safe Chains

Frontier-safe chains are built in two stages. First, each row picks its unique
deepest predecessor as a candidate scan parent. Rows with tied deepest
predecessors are counted as unresolved. Second, every candidate chain is split
by the chain-head boundary-known rule: when extending `current -> child`, every
non-main predecessor of `child` except `current` must be smaller than the
current chain head. Otherwise the chain is cut before `child`, and `child`
becomes a new chain head.

frontier-safe 链现在分两步。第一步，每一行用“唯一最深前驱”生成候选 scan parent；如果存在多个并列最深前驱，则该行记为 unresolved。第二步，对候选链按“链头边界已知”规则切链：从 `current` 扩展到 `child` 时，`child` 除 `current` 之外的所有非主依赖都必须小于当前链头 `head`；否则在 `child` 前切链，并让 `child` 作为新链头继续。

```bash
python3 scripts/analyze_scan_chains.py \
  data/factors/spcg/tmt_sym/spcg_spilu_l_tmt_sym_fill10_dropdefault_sp0.mtx \
  --source factor-l \
  --frontier-safe
```

Default output:

默认输出：

```text
results/frontier_safe/spcg_factor_l.csv
```

当前真实 `L` 因子 `tmt_sym` 的结果：

```text
strict coverage:               2.70%
frontier-safe eligible rows:   96.56%
frontier-safe boundary cuts:   307787
frontier-safe coverage:        92.64%
max frontier-safe chain length: 61
```

## Regression

## 回归测试

Small regression matrices live in:

小测试矩阵位置：

```text
data/matrices/regression/
```

Run strict regression:

运行 strict 小测试：

```bash
python3 scripts/analyze_scan_chains.py --small-test
```

Run frontier-safe regression:

运行 frontier-safe 小测试：

```bash
python3 scripts/analyze_scan_chains.py --small-test --frontier-safe
```

Regression outputs:

回归测试输出：

```text
results/regression/strict_small_latest.csv
results/regression/frontier_safe_small_latest.csv
```

## Drop-Element Experiments

## 删元素扫描实验

### Preparing SuiteSparse L Factors

### 准备 SuiteSparse 真实 L 因子

To expand beyond the single existing `tmt_sym` factor, use the SuiteSparse
preparation entry point. It downloads a curated SPD / structural / thermal set
and generates SPCG-style SuperLU `L/U` factors.

为了把实验对象从单个 `tmt_sym` 扩展到 10-20 个真实因子，可以使用 SuiteSparse 准备脚本。它会下载筛选好的 SPD / 结构 / 热问题矩阵，并生成 SPCG 风格 SuperLU `L/U` 因子。

Recommended first batch:

推荐先跑 12 个矩阵：

```bash
python3 scripts/prepare_suitesparse_factors.py --limit 12
```

List all curated candidates:

查看全部内置候选：

```bash
python3 scripts/prepare_suitesparse_factors.py --list
```

Prepare CFD / nonsymmetric PDE matrices for GMRES/ILU tests:

准备 CFD / 非对称 PDE 矩阵，用于 GMRES/ILU 场景：

```bash
python3 scripts/prepare_suitesparse_factors.py \
  --candidate-set cfd \
  --permc-spec COLAMD
```

Outputs:

输出位置：

```text
data/matrices/real/<matrix>/<matrix>.mtx
data/factors/spcg/<matrix>/spcg_spilu_l_<matrix>_fill10_drop0.0001_sp0.mtx
data/factors/spcg/<matrix>/spcg_spilu_u_<matrix>_fill10_drop0.0001_sp0.mtx
results/factor_preparation/suitesparse_spd_structural_thermal_manifest.csv
results/factor_preparation/suitesparse_cfd_nonsym_pde_manifest.csv
```

The manifest records download status, factorization status, Matrix Market
metadata, factor paths, and failure reasons. A failed matrix does not stop the
rest of the batch.

manifest 会记录下载状态、分解状态、Matrix Market 元数据、因子路径和失败原因。某个矩阵失败不会中断整批任务。

For details:

详细说明见：

```text
docs/SUITESPARSE_FACTOR_PREPARATION.md
```

The next-stage static experiment asks whether deleting a small number of
low-risk entries from real `L` factors can create more useful scan chains. The
main entry point is:

下一阶段静态实验用于判断：对真实 `L` 因子删除少量低风险元素，是否能制造更多有执行价值的 scan 链。入口是：

```bash
python3 scripts/run_drop_experiment.py data/factors/spcg \
  --source factor-l
```

It supports `no-drop`, `magnitude-drop`, `wavefront-oriented`,
`scan-chain-aware`, and `long-chain-aware-drop` strategies, with default drop
ratios `1%`, `2%`, `5%`, and `10%`.

当前支持 `no-drop`、`magnitude-drop`、`wavefront-oriented`、`scan-chain-aware`、`long-chain-aware-drop` 五种策略，默认扫描 `1%`、`2%`、`5%`、`10%` 删除比例。当前主要判断标准不是 strict 总覆盖率，而是长度 `>=8/>=16` 的有效长链覆盖率增量。

Default outputs:

默认输出：

```text
results/drop_experiments/drop_scan_chain_sweep.csv
results/drop_experiments/drop_scan_chain_summary.txt
results/definitions/drop_experiment_fields.csv
figures/drop_experiments/
```

Current long-chain batch output:

当前长链导向批量结果：

```text
results/drop_experiments/long_chain_factor_batch.csv
results/drop_experiments/long_chain_factor_batch_summary.txt
figures/drop_experiments/long_chain_factor_batch/
```

Formal 14 frontier/relaxed bounded-width region experiment:

正式 14 因子的 frontier/relaxed bounded-width region 结构裁决实验，中文简称“链变块”：

```bash
python3 scripts/run_formal_14_frontier_relaxed_region_experiment.py
```

This experiment does not drop entries, download matrices, regenerate factors,
or run a GPU solver. It checks whether existing ILU `L` factors contain enough
bounded-width scan regions for `w=1,2,4,8`.

该实验不删边、不下载新矩阵、不重新生成因子，也不运行 GPU solver。它只检查已有 ILU `L`
因子中是否存在足够的 bounded-width scan region，测试 `w=1,2,4,8`。
“链变块”的含义是：在 strict 单前驱长链失败后，不再强求一条线性 scan 链，而是把候选执行单元放宽成
bounded-width scan region，也就是带少量 live internal dependencies 的连续行块。

Formal frontier/relaxed output:

正式 frontier/relaxed 输出：

```text
results/drop_experiments/formal_14_frontier_relaxed_region.csv
results/drop_experiments/formal_14_frontier_relaxed_region_summary.txt
results/drop_experiments/formal_14_frontier_relaxed_region_spd_summary.txt
results/drop_experiments/formal_14_frontier_relaxed_region_cfd_summary.txt
figures/drop_experiments/formal_14_frontier_relaxed_region/
```

The result CSV uses upsert by default: matching matrix/source/path/method/parameter
rows are updated, while unrelated old rows are kept. Use `--replace-out` to
rewrite the file from only the current run.

结果 CSV 默认按“同矩阵、同来源、同路径、同方法、同参数组合”更新已有行，不相关旧行保留。若只想保留本次运行结果，使用 `--replace-out`。

For full method details and commands, see:

完整方法说明和命令见：

```text
docs/DROP_ELEMENT_SCAN_EXPERIMENT.md
```

## Current Result Files

## 当前主要结果文件

```text
results/strict/matrix_lower.csv
results/strict/spcg_factor_l.csv
results/frontier_safe/matrix_lower.csv
results/frontier_safe/spcg_factor_l.csv
results/drop_experiments/drop_scan_chain_sweep.csv
results/drop_experiments/drop_scan_chain_summary.txt
results/drop_experiments/long_chain_factor_batch.csv
results/drop_experiments/long_chain_factor_batch_summary.txt
results/drop_experiments/formal_14_frontier_relaxed_region.csv
results/drop_experiments/formal_14_frontier_relaxed_region_summary.txt
results/drop_experiments/formal_14_frontier_relaxed_region_spd_summary.txt
results/drop_experiments/formal_14_frontier_relaxed_region_cfd_summary.txt
results/definitions/strict_fields.csv
results/definitions/frontier_safe_fields.csv
results/definitions/drop_experiment_fields.csv
results/factor_preparation/suitesparse_spd_structural_thermal_manifest.csv
```

中文说明：

- `results/strict/matrix_lower.csv`：原矩阵直接抽下三角后的 strict 链统计。
- `results/strict/spcg_factor_l.csv`：真实 SPCG `L` 因子的 strict 链统计。
- `results/frontier_safe/matrix_lower.csv`：原矩阵下三角的 strict + frontier-safe 统计。
- `results/frontier_safe/spcg_factor_l.csv`：真实 SPCG `L` 因子的 strict + frontier-safe 统计。
- `results/drop_experiments/drop_scan_chain_sweep.csv`：删元素扫描实验总表。
- `results/drop_experiments/drop_scan_chain_summary.txt`：删元素扫描实验自动总结。
- `results/drop_experiments/long_chain_factor_batch.csv`：长链导向删元素批量实验总表。
- `results/drop_experiments/long_chain_factor_batch_summary.txt`：长链导向实验最终判断摘要。
- `results/drop_experiments/formal_14_frontier_relaxed_region.csv`：正式 14 因子的 bounded-width relaxed region 结构裁决总表。
- `results/drop_experiments/formal_14_frontier_relaxed_region_summary.txt`：正式 14 因子的 frontier/relaxed region 总裁决。
- `results/drop_experiments/formal_14_frontier_relaxed_region_spd_summary.txt`：SPD / 结构 / 热问题组裁决。
- `results/drop_experiments/formal_14_frontier_relaxed_region_cfd_summary.txt`：CFD / 非对称 PDE 组裁决。
- `results/definitions/strict_fields.csv`：strict 字段定义。
- `results/definitions/frontier_safe_fields.csv`：frontier-safe 追加字段定义。
- `results/definitions/drop_experiment_fields.csv`：删元素实验字段定义。
- `results/factor_preparation/suitesparse_spd_structural_thermal_manifest.csv`：SuiteSparse 下载和 SPCG `L/U` 因子生成记录。

## Notes

## 注意事项

Large `.mtx` files and generated factors are intentionally ignored by git. The
directory structure is kept with `.gitkeep` files, while result CSVs are kept in
`results/`.

大矩阵 `.mtx` 和生成的真实 `L/U` 因子默认不进入 git，避免提交超大文件。目录结构通过 `.gitkeep` 保留，统计结果 CSV 保存在 `results/` 下。

For the exact list of local-only files and data placeholders, see:

本地保留但不上传 GitHub 的文件清单和数据目录占位说明见：

```text
docs/NOT_UPLOADED_TO_GITHUB.md
```

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

## Current Result Files

## 当前主要结果文件

```text
results/strict/matrix_lower.csv
results/strict/spcg_factor_l.csv
results/frontier_safe/matrix_lower.csv
results/frontier_safe/spcg_factor_l.csv
results/definitions/strict_fields.csv
results/definitions/frontier_safe_fields.csv
```

中文说明：

- `results/strict/matrix_lower.csv`：原矩阵直接抽下三角后的 strict 链统计。
- `results/strict/spcg_factor_l.csv`：真实 SPCG `L` 因子的 strict 链统计。
- `results/frontier_safe/matrix_lower.csv`：原矩阵下三角的 strict + frontier-safe 统计。
- `results/frontier_safe/spcg_factor_l.csv`：真实 SPCG `L` 因子的 strict + frontier-safe 统计。
- `results/definitions/strict_fields.csv`：strict 字段定义。
- `results/definitions/frontier_safe_fields.csv`：frontier-safe 追加字段定义。

## Notes

## 注意事项

Large `.mtx` files and generated factors are intentionally ignored by git. The
directory structure is kept with `.gitkeep` files, while result CSVs are kept in
`results/`.

大矩阵 `.mtx` 和生成的真实 `L/U` 因子默认不进入 git，避免提交超大文件。目录结构通过 `.gitkeep` 保留，统计结果 CSV 保存在 `results/` 下。

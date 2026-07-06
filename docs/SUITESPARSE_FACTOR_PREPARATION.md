# SuiteSparse SPD / Structural / Thermal Factor Preparation

# SuiteSparse SPD / 结构 / 热问题因子准备

This document describes how to prepare 10-20 real `L` factors for the
scan-chain drop-element experiments.

本文档说明如何准备 10-20 个真实 `L` 因子，用于后续 scan-chain 删元素实验。

## Goal

## 目标

The current drop-element framework needs more real preconditioner-like `L`
factors. The target data class is:

当前删元素框架需要更多接近预条件器场景的真实 `L` 因子。目标数据类型为：

```text
SPD / structural / thermal problems
10-20 matrices
PCG / IC / ILU scenario
```

The preparation script downloads selected SuiteSparse Matrix Collection
matrices and generates SPCG-style SuperLU `L/U` factors. SciPy does not provide
a production incomplete-Cholesky implementation here, so the generated factors
are ILU-style `spilu` factors used as the current proxy for PCG/IC/ILU static
dependency analysis.

准备脚本会从 SuiteSparse Matrix Collection 下载筛选矩阵，并生成 SPCG 风格 SuperLU `L/U` 因子。当前环境没有可直接使用的生产级 incomplete Cholesky，因此这里用 `spilu` 生成 ILU 风格因子，作为 PCG/IC/ILU 静态依赖分析的代理对象。

## Entry Point

## 入口

```bash
python3 scripts/prepare_suitesparse_factors.py --limit 12
```

This first batch includes a thermal problem, an SPD/FEM problem, and structural
stiffness matrices. It is the recommended starting point before running all 20
candidates.

第一批推荐 `--limit 12`，会覆盖热问题、SPD/FEM 对照和结构刚度矩阵。完整 20 个候选稳定后再扩大运行。

List candidates:

查看候选：

```bash
python3 scripts/prepare_suitesparse_factors.py --list
```

Run selected matrices only:

只跑指定矩阵：

```bash
python3 scripts/prepare_suitesparse_factors.py \
  --only thermal1,bcsstk13,bcsstk16
```

Run all curated candidates:

运行全部内置候选：

```bash
python3 scripts/prepare_suitesparse_factors.py
```

Run the CFD / nonsymmetric PDE set for GMRES/ILU experiments:

运行 CFD / 非对称 PDE 集合，用于 GMRES/ILU 实验：

```bash
python3 scripts/prepare_suitesparse_factors.py \
  --candidate-set cfd \
  --permc-spec COLAMD
```

`COLAMD` is used for this set because some nonsymmetric PDE matrices are
singular under `NATURAL` SuperLU ILU ordering. The generated factor filename
includes `_permccolamd_` so these factors are not confused with the default
SPD/structural factors.

该集合使用 `COLAMD`，因为部分非对称 PDE 矩阵在 `NATURAL` SuperLU ILU 排序下会 exact singular。生成的因子文件名包含 `_permccolamd_`，避免和默认 SPD/结构因子混淆。

## Outputs

## 输出

Downloaded matrices:

下载后的原矩阵：

```text
data/matrices/real/<matrix>/<matrix>.mtx
```

Generated factors:

生成的因子：

```text
data/factors/spcg/<matrix>/spcg_spilu_l_<matrix>_fill10_drop0.0001_sp0.mtx
data/factors/spcg/<matrix>/spcg_spilu_u_<matrix>_fill10_drop0.0001_sp0.mtx
```

Manifest:

运行记录：

```text
results/factor_preparation/suitesparse_spd_structural_thermal_manifest.csv
results/factor_preparation/suitesparse_cfd_nonsym_pde_manifest.csv
```

The manifest is rewritten during each run and updated after every matrix. It
contains the SuiteSparse source, Matrix Market metadata, generated factor paths,
download/factorization status, timing, and failure reason.

manifest 每次运行重写，并在每个矩阵结束后立即更新。它记录 SuiteSparse 来源、Matrix Market 元数据、生成因子路径、下载/分解状态、耗时和失败原因。

## Default Factorization Parameters

## 默认分解参数

```text
factor_method:      spilu
fill_factor:        10
drop_tol:           1e-4
diagonal_shift:     0
permc_spec:         NATURAL
diag_pivot_thresh:  1e-15
```

`NATURAL` is used by default because the scan-chain analysis should reflect the
input ordering rather than a hidden fill-reducing permutation. If too many
matrices fail during factorization, use `--permc-spec COLAMD` as a diagnostic
variant, but keep its results separate.

默认使用 `NATURAL`，因为 scan-chain 分析应反映输入排序下的依赖结构，而不是被隐藏的填充优化重排序改变。如果大量矩阵分解失败，可以用 `--permc-spec COLAMD` 做诊断对照，但结果应和默认结果分开。

## Curated Candidate Set

## 内置候选集

The script currently contains up to 20 candidates:

脚本当前内置最多 20 个候选：

```text
Schmid/thermal1
GHS_psdef/apache1
HB/bcsstk06
HB/bcsstk07
HB/bcsstk08
HB/bcsstk09
HB/bcsstk10
HB/bcsstk11
HB/bcsstk12
HB/bcsstk13
HB/bcsstk14
HB/bcsstk15
HB/bcsstk16
HB/bcsstk17
HB/bcsstk18
HB/bcsstk03
HB/bcsstk04
HB/bcsstk05
Schmid/thermal2
GHS_psdef/apache2
```

The ordering intentionally puts a diverse, moderate first batch before the
larger `thermal2` and `apache2` matrices.

这个顺序刻意把多样化、中等规模的第一批放在前面，把更大的 `thermal2` 和 `apache2` 放在后面。

The CFD / nonsymmetric PDE set currently contains 6 GMRES/ILU candidates:

CFD / 非对称 PDE 集合当前包含 6 个 GMRES/ILU 候选：

```text
Rothberg/cfd1
Rothberg/cfd2
FIDAP/ex11
FIDAP/ex19
FIDAP/ex15
Simon/raefsky3
```

## Useful Options

## 常用参数

Limit archive size when testing:

测试时限制下载大小：

```bash
python3 scripts/prepare_suitesparse_factors.py \
  --limit 20 \
  --max-download-mb 100
```

Download without factorization:

只下载不分解：

```bash
python3 scripts/prepare_suitesparse_factors.py --limit 12 --download-only
```

Regenerate existing factors:

重新生成已有因子：

```bash
python3 scripts/prepare_suitesparse_factors.py \
  --only bcsstk13 \
  --overwrite-factors
```

Try a diagnostic reordered factorization:

尝试重排序诊断分解：

```bash
python3 scripts/prepare_suitesparse_factors.py \
  --only bcsstk13 \
  --permc-spec COLAMD \
  --overwrite-factors
```

## Next Step After Factor Generation

## 生成因子后的下一步

Run the long-chain-aware drop experiment over all generated `L` factors:

生成因子后，对全部 `L` 因子运行长链导向删元素实验：

```bash
python3 scripts/run_drop_experiment.py data/factors/spcg \
  --source factor-l \
  --methods no-drop,magnitude-drop,wavefront-oriented,scan-chain-aware,long-chain-aware-drop \
  --drop-ratios 0.01,0.02,0.05,0.10 \
  --risk-thresholds 0.0001,0.001,0.01 \
  --max-drop-per-row 1,-1 \
  --replace-out \
  --out results/drop_experiments/suitesparse_long_chain_batch.csv \
  --summary-out results/drop_experiments/suitesparse_long_chain_batch_summary.txt \
  --fig-dir figures/drop_experiments/suitesparse_long_chain_batch
```

The decision criterion remains strict: do not judge success by total strict
chain coverage. Judge by stable gains in length `>=8` and `>=16` effective
chain coverage across multiple real `L` factors.

判断标准仍然严格：不要看 strict 总覆盖率是否增加，而要看多个真实 `L` 因子上长度 `>=8` 和 `>=16` 的有效长链覆盖率是否稳定提升。

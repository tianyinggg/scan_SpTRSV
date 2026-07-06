# Not Uploaded To GitHub

# 未上传 GitHub 的本地内容说明

This repository intentionally does not upload large Matrix Market inputs,
generated factor matrices, or Python cache files. The code, scripts, result
CSVs, summaries, and figures are tracked in Git; the heavy reproducibility
inputs remain local.

本仓库有意不上传大规模 Matrix Market 输入矩阵、生成的因子矩阵和 Python 缓存文件。代码、脚本、结果 CSV、summary 和图表进入 Git；大体量复现实验输入保留在本地。

## Current Branch Status

## 当前分支状态

```text
branch: delete-element
latest local commit: 081f81d Add delete-element long-chain drop experiments
remote: git@github.com:tianyinggg/scan_SpTRSV.git
```

At the time this note was written, `delete-element` is the active branch. Use
the following command to confirm whether the local branch is already synced:

写入本文档时，当前活动分支是 `delete-element`。可用下面命令确认本地分支是否已经和远端同步：

```bash
git status --short --branch
```

If it shows `## delete-element...origin/delete-element` with no modified or
untracked tracked files, the uploadable part is synced. If it shows `ahead`,
push with:

如果输出为 `## delete-element...origin/delete-element` 且没有可跟踪文件变更，说明可上传部分已经同步。如果显示 `ahead`，执行：

```bash
git push -u origin delete-element
```

## Why These Files Are Not Uploaded

## 为什么这些文件不上传

The repository `.gitignore` excludes:

仓库 `.gitignore` 排除了：

```text
*.mtx
*.bin
*.tmp
*.log
*.pyc
__pycache__/
data/matrices/**/*.mtx
data/factors/**/*.mtx
```

Reason:

原因：

- Matrix files and generated factors are large binary/text data products and should not be stored directly in normal Git history.
- SuiteSparse inputs can be downloaded again with `scripts/prepare_suitesparse_factors.py`.
- Generated `L/U` factors can be regenerated from the manifests and scripts.
- Python `__pycache__` files are machine-local build artifacts.

## Local Data Not Uploaded

## 未上传的本地数据

Approximate local size:

本地大致规模：

```text
data/matrices/real/**/*.mtx      about 10 GB
data/factors/spcg/**/*.mtx       about 3.9 GB
data/matrices/regression/*.mtx   about 20 KB
total ignored .mtx data          about 14 GB
```

### Real Matrix Inputs

### 真实矩阵输入

Formal 14 dataset names:

正式 14 个数据集名称：

```text
thermal1
apache1
bcsstk10
bcsstk13
bcsstk15
bcsstk16
bcsstk17
bcsstk18
cfd1
cfd2
ex11
ex19
ex15
raefsky3
```

Additional local matrix names:

额外本地矩阵名称：

```text
FullChip
kkt_power
mawi_201512020030
nlpkkt200
tmt_sym
vas_stokes_4M
```

These directories contain local `.mtx` files that are ignored by Git:

以下目录包含本地 `.mtx` 文件，不进入 Git：

```text
data/matrices/real/FullChip/
data/matrices/real/apache1/
data/matrices/real/bcsstk10/
data/matrices/real/bcsstk13/
data/matrices/real/bcsstk15/
data/matrices/real/bcsstk16/
data/matrices/real/bcsstk17/
data/matrices/real/bcsstk18/
data/matrices/real/cfd1/
data/matrices/real/cfd2/
data/matrices/real/ex11/
data/matrices/real/ex15/
data/matrices/real/ex19/
data/matrices/real/kkt_power/
data/matrices/real/mawi_201512020030/
data/matrices/real/nlpkkt200/
data/matrices/real/raefsky3/
data/matrices/real/thermal1/
data/matrices/real/tmt_sym/
data/matrices/real/vas_stokes_4M/
```

### Generated SPCG/ILU Factors

### 生成的 SPCG/ILU 因子

Generated factor dataset names:

生成因子数据集名称：

```text
thermal1
apache1
bcsstk10
bcsstk13
bcsstk15
bcsstk16
bcsstk17
bcsstk18
cfd1
cfd2
ex11
ex19
ex15
raefsky3
tmt_sym
```

These directories contain generated local `L/U` factor `.mtx` files that are
ignored by Git:

以下目录包含生成的本地 `L/U` 因子 `.mtx` 文件，不进入 Git：

```text
data/factors/spcg/apache1/
data/factors/spcg/bcsstk10/
data/factors/spcg/bcsstk13/
data/factors/spcg/bcsstk15/
data/factors/spcg/bcsstk16/
data/factors/spcg/bcsstk17/
data/factors/spcg/bcsstk18/
data/factors/spcg/cfd1/
data/factors/spcg/cfd2/
data/factors/spcg/ex11/
data/factors/spcg/ex15/
data/factors/spcg/ex19/
data/factors/spcg/raefsky3/
data/factors/spcg/thermal1/
data/factors/spcg/tmt_sym/
```

### Regression Matrix Inputs

### 回归测试矩阵输入

The small regression `.mtx` files are also ignored by the global `*.mtx` rule:

小回归测试 `.mtx` 也被全局 `*.mtx` 规则忽略：

```text
data/matrices/regression/t01_two_short_chains_4.mtx
data/matrices/regression/t02_pure_chain_8.mtx
data/matrices/regression/t03_invalid_same_level_4.mtx
data/matrices/regression/t04_external_dep_chain_8.mtx
data/matrices/regression/t05_pure_chain_40.mtx
```

These are small enough to version if desired, but currently remain ignored for
policy consistency. If they should be uploaded later, use a targeted `.gitignore`
exception.

这些文件很小，如果之后需要也可以纳入版本管理；当前为了策略一致仍保持忽略。若后续要上传，应在 `.gitignore` 中为这些文件添加定向例外。

## Uploaded Reproducibility Metadata

## 已上传的复现实验元数据

The following files are tracked and describe how to recreate the local data:

以下文件会进入 Git，用于说明如何重建本地数据：

```text
docs/SUITESPARSE_FACTOR_PREPARATION.md
results/factor_preparation/suitesparse_spd_structural_thermal_manifest.csv
results/factor_preparation/suitesparse_cfd_nonsym_pde_manifest.csv
scripts/prepare_suitesparse_factors.py
scripts/run_formal_14_long_chain_experiment.py
```

Formal experiment outputs are uploaded:

正式实验结果会上传：

```text
results/drop_experiments/formal_14_long_chain_batch.csv
results/drop_experiments/formal_14_long_chain_batch_summary.txt
results/drop_experiments/formal_14_spd_summary.txt
results/drop_experiments/formal_14_cfd_summary.txt
figures/drop_experiments/formal_14_long_chain_batch/
```

## Recreate Local Data

## 重建本地数据

Prepare the formal SuiteSparse factors:

重建正式 SuiteSparse 因子：

```bash
python3 scripts/prepare_suitesparse_factors.py \
  --only thermal1,apache1,bcsstk10,bcsstk13,bcsstk15,bcsstk16,bcsstk17,bcsstk18

python3 scripts/prepare_suitesparse_factors.py \
  --candidate-set cfd \
  --permc-spec COLAMD
```

Run the formal decision experiment:

运行正式裁决实验：

```bash
python3 scripts/run_formal_14_long_chain_experiment.py
```

## Placeholder Policy

## 占位文件策略

Data directories contain `README.md` placeholders so that GitHub preserves the
intended directory structure even though `.mtx` files are ignored.

数据目录中放置 `README.md` 占位文件，因此即使 `.mtx` 文件被忽略，GitHub 也能保留预期目录结构并说明每个目录用途。

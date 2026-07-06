# Real Matrix Inputs

This directory contains real SuiteSparse and large benchmark Matrix Market
inputs. The `.mtx` files are ignored by Git. Per-matrix subdirectories contain
README placeholders to preserve the expected layout on GitHub.

本目录包含 SuiteSparse 和其他大型真实矩阵输入。`.mtx` 文件不上传 Git。每个矩阵子目录中
放置 README 占位文件，用于在 GitHub 上保留目录结构。

## Formal 14 Matrix Inputs

## 正式 14 个矩阵输入

SPD / structural / thermal group:

SPD / 结构 / 热问题组：

```text
thermal1
apache1
bcsstk10
bcsstk13
bcsstk15
bcsstk16
bcsstk17
bcsstk18
```

CFD / nonsymmetric PDE group:

CFD / 非对称 PDE 组：

```text
cfd1
cfd2
ex11
ex19
ex15
raefsky3
```

## Additional Local Matrices

## 其他本地矩阵

```text
FullChip
kkt_power
mawi_201512020030
nlpkkt200
tmt_sym
vas_stokes_4M
```

These files are local-only:

这些文件仅在本地保存：

```text
data/matrices/real/<matrix>/<matrix>.mtx
```

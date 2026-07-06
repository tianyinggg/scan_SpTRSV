# SPCG / ILU Factor Directory

This directory stores generated SuperLU `L/U` factors for SPCG/ILU-style
static experiments. Factor `.mtx` files are ignored by Git; subdirectory
README placeholders document the expected local files.

本目录保存 SPCG/ILU 风格静态实验生成的 SuperLU `L/U` 因子。因子 `.mtx` 文件不上传 Git；
子目录 README 占位文件记录本地应存在的文件。

## Formal 14 Generated Factors

## 正式 14 个生成因子

SPD / structural / thermal factors generated with `spilu`, `drop_tol=1e-4`,
and `permc_spec=NATURAL`:

SPD / 结构 / 热问题因子，使用 `spilu`、`drop_tol=1e-4`、`permc_spec=NATURAL`：

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

CFD / nonsymmetric PDE factors generated with `spilu`, `drop_tol=1e-4`, and
`permc_spec=COLAMD`:

CFD / 非对称 PDE 因子，使用 `spilu`、`drop_tol=1e-4`、`permc_spec=COLAMD`：

```text
cfd1
cfd2
ex11
ex19
ex15
raefsky3
```

Additional local factor directory:

额外本地因子目录：

```text
tmt_sym
```

Expected local files:

本地应存在的文件形态：

```text
data/factors/spcg/<matrix>/spcg_spilu_l_<matrix>_*.mtx
data/factors/spcg/<matrix>/spcg_spilu_u_<matrix>_*.mtx
```

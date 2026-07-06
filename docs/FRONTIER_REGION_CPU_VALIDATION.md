# Frontier Region CPU 数值验证

本文档说明正式 14 个 `L` 因子的最后一轮 CPU 数值验证。目标不是做 GPU kernel，也不是继续删边，而是回答一个更基础的问题：

```text
frontier-safe / bounded-width relaxed region
是否真的可以在不改结果的前提下，等价转成小状态 affine scan 执行？
```

这一步承接“链变块”结构实验：

- strict 单前驱长链已经证明不成立。
- bounded-width relaxed region 结构实验显示 `w=4` 下存在一批有覆盖率的矩阵。
- 现在只验证这些 region 在 CPU 上是否能精确重放，不讨论性能。

## 新增入口

- `scripts/run_formal_14_frontier_region_cpu_validation.py`
- `src/scan_sptrsv/formal_14_frontier_region_cpu_validation.py`

运行命令：

```bash
python3 scripts/run_formal_14_frontier_region_cpu_validation.py
```

默认输出：

```text
results/drop_experiments/formal_14_frontier_region_cpu_validation.csv
results/drop_experiments/formal_14_frontier_region_cpu_validation_summary.txt
```

## 输入范围

只使用 formal_14 正式数据集，不下载新矩阵，不修改因子生成逻辑：

SPD / 结构 / 热问题：

- `thermal1`
- `apache1`
- `bcsstk10`
- `bcsstk13`
- `bcsstk15`
- `bcsstk16`
- `bcsstk17`
- `bcsstk18`

CFD / 非对称 PDE：

- `cfd1`
- `cfd2`
- `ex11`
- `ex19`
- `ex15`
- `raefsky3`

## 验证流程

### 1. CPU baseline SpTRSV

对每个正式 `L` 因子：

- 生成固定随机种子的 `x_true`
- 计算 `b = L * x_true`
- 用普通 CPU 顺序前代求 `x_ref`
- 记录：
  - `baseline_max_abs_error = max |x_ref - x_true|`
  - `baseline_rel_residual = ||L x_ref - b|| / ||b||`

这一步不是为了追求所有矩阵都有很小的 `x_ref - x_true`，而是为了得到可比较的 CPU 参考解。某些矩阵如果本身病态，`x_ref` 仍可作为“等价到 baseline”的验证对象。

### 2. frontier-safe 检查

结构实验先生成 bounded-width region。这里再检查每个 region：

- region 外依赖是否都在 region 之前
- region 内依赖是否都来自更早的 region 内行

若存在未解析依赖，则该 region 记为失败，不进入后续仿真。

### 3. region 顺序重放

把每个 region 的依赖拆成：

- `external deps`
- `internal deps`

先扣除 external 贡献：

```text
b'_i = b_i - sum_external L[i,j] * x_ref[j]
```

再在 region 内部按顺序前代：

```text
x_i = (b'_i - sum_internal L[i,j] * x_j) / L[i,i]
```

若这一层都不通过，说明 external/internal 划分本身有问题。

### 4. 小状态 affine 转移验证

对 width=`w` 的 region，只保留最多 `w` 个 live internal values 作为状态 `S_t`。

每处理一行，构造：

```text
S_{t+1} = A_t * S_t + c_t
x_row = p_t^T * S_t + q_t
```

CPU 上顺序应用这些仿射转移，得到 `x_region_affine`，再对比 `x_ref`。

### 5. prefix scan 模拟

把每一步转移记作：

```text
T_t = (A_t, c_t)
```

验证仿射复合律：

```text
(A2, c2) o (A1, c1) = (A2*A1, A2*c1 + c2)
```

然后在 CPU 上做 prefix-compose scan 模拟，恢复每一步状态和每行输出，得到 `x_region_scan_sim`，最后对比 `x_ref`。

## 当前验证宽度

本轮只验证：

```text
w = 2, 4
```

因为结构实验真正有判断意义的是 `w<=4`。若只有 `w=8` 才有效，执行状态成本就已经偏大，不足以支撑主线。

## CSV 关键字段

- `matrix` / `group` / `w`
- `region_count`
- `frontier_safe_region_count`
- `validated_region_count`
- `failed_region_count`
- `covered_rows`
- `covered_row_ratio`
- `structural_ge8_row_coverage`
- `coverage_retention`
- `baseline_max_abs_error`
- `baseline_rel_residual`
- `max_abs_error_seq`
- `max_abs_error_affine`
- `max_abs_error_scan_sim`
- `max_rel_error_scan_sim`
- `external_unresolved_count`
- `avg_external_nnz_per_row`
- `avg_internal_nnz_per_row`
- `verdict`

其中最关键的解释是：

- `coverage_retention = validated_coverage / structural_ge8_coverage`
- 若 `coverage_retention` 明显小于 1，说明结构实验过于乐观。
- 若 `seq` 通过但 `affine/scan_sim` 不通过，说明 region 虽能顺序算，但不能自然转成 scan。

## 当前裁决规则

summary 当前使用的继续条件：

- `w=4` 支持矩阵至少 `8/9` 通过
- CFD 组 `w=4` 支持矩阵至少 `4/5` 通过
- `max_abs_error_scan_sim <= 1e-9`
  或与 CPU baseline 误差同量级
- `coverage_retention >= 0.90`

若满足，则当前结论写为：

```text
Final decision: B. CPU validation supports bounded-width scan-region equivalence;
next step should be CPU executor / numerical convergence study, not GPU kernel.
```

## 当前结果如何解释

这次验证回答的是“能不能等价表达”，不是“值不值得实现高性能 executor”。

- 如果 `max_abs_error_seq / affine / scan_sim` 都与 `x_ref` 一致，说明 region 的结构抽象没有错。
- 如果 `external_unresolved_count = 0` 且 `coverage_retention ≈ 1`，说明 frontier-safe 过滤没有把结构覆盖率大幅吃掉。
- 如果个别矩阵 baseline 本身病态，例如 `x_ref` 对 `x_true` 的误差大，summary 会单独给出 baseline stability diagnostics。这种情况只能说明：
  - affine / scan_sim 与 CPU baseline 等价；
  - 不能单独据此宣称该矩阵的数值求解本身很健康。

## 这一步之后该做什么

如果 summary 仍保持支持结论，下一步应是：

- 先做 CPU executor 级别的数值收敛与实现可行性验证
- 再判断是否值得进入并行 prefix / GPU executor

如果未来批量验证发现：

- 需要更宽的 `w`
- 覆盖集中在少数矩阵
- 或数值稳定性依赖过强的前提

则应重新收窄场景，或者终止 scan-region 主线。

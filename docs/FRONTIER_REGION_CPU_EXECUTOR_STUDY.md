# Frontier Region CPU Executor Study

本文档记录 formal_14 正式 14 个 `L` 因子的最后一轮 CPU executor / cost-model 裁决实验。

背景链路已经很明确：

1. strict 单前驱长链失败；
2. 删元素制造 strict 长链失败；
3. bounded-width frontier-safe region 在结构上成立；
4. 这些 region 在 CPU 数值验证上与 baseline 等价；
5. 现在只剩最后一个问题：

```text
就算这些 region 能精确计算，执行成本是否足以支持继续做 scan-region executor？
```

这一步仍然：

- 不下载新矩阵；
- 不改因子生成逻辑；
- 不删边；
- 不写 GPU kernel；
- 不做激进性能优化。

目标只是给出一份可信的 CPU 侧成本分解和方向裁决。

## 新增入口

- `scripts/run_formal_14_frontier_region_cpu_executor_study.py`
- `src/scan_sptrsv/formal_14_frontier_region_cpu_executor_study.py`

运行方式：

```bash
python3 scripts/run_formal_14_frontier_region_cpu_executor_study.py
```

默认输出：

```text
results/drop_experiments/formal_14_frontier_region_cpu_executor_study.csv
results/drop_experiments/formal_14_frontier_region_cpu_executor_study_summary.txt
figures/drop_experiments/formal_14_frontier_region_cpu_executor_study/
```

## 实验对象

继续只使用 formal_14 的正式因子：

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

## 研究的 4 条执行路径

### 1. `baseline_cpu_sptrsv`

普通 CPU 顺序前代，作为正确性和时间参考。

### 2. `region_seq_replay`

只对有效 region 做 frontier-safe external reduction，region 内部仍然顺序求解。

它的作用是隔离“region 化”本身带来的额外成本。

### 3. `region_affine_executor`

把 region 编译成 bounded-width 小状态转移，CPU 上顺序应用 affine transitions。

它衡量的是：

- 小状态表示是否可落地；
- metadata 和状态搬运成本是否明显偏大。

### 4. `region_scan_sim_executor`

不做真正并行 scan，但在 CPU 上顺序模拟 prefix-compose 所需的工作和状态恢复。

它主要服务于未来 GPU executor 的理论工作量估计，而不是追求当前 CPU 更快。

## 当前只执行哪些 region

本实验只把：

```text
region_len >= 8
```

的 frontier-safe bounded-width region 视为有效候选执行单元。

这样做是为了避免把长度 2/3/4 的小块混进覆盖率，造成“看上去很多、实际上不值执行”的假象。

因此：

- `covered_rows` 在本实验里等同于 `ge8_covered_rows`
- `ge16_covered_rows` 仍然单独统计

## 运行宽度

本轮默认跑：

```text
w = 2, 4, 8
```

其中：

- `w=2`、`w=4` 是真正关心的范围；
- `w=8` 只是上界参考；
- 如果只有 `w=8` 才有大覆盖，通常已经意味着状态和 compose 成本太高，不足以支撑主线。

## 成本模型口径

这套成本模型刻意保持保守和可解释，不追求精致到像硬件模型。

### baseline work

```text
baseline_est_ops = strict_lower_nnz + n
```

解释：

- 每条 strict-lower 边记一次依赖访问/归约工作；
- 每行再记一次求解步。

### external reduction work

```text
external_reduction_est_ops = external_nnz_total
```

解释：

- region 外依赖必须先归约掉；
- external 依赖过多会直接吃掉 scan-region 的结构收益。

### affine transition work

对每个 covered row：

```text
2 * active_state_inputs + 1
```

解释：

- 只统计行输出 `x_row = p^T S + q` 的稀疏小状态算术；
- 不把纯 metadata 搬运算到这里，metadata 单独计。

### scan compose work

每个 region 用一个保守的 upsweep+downsweep 近似：

```text
scan_compose_est_ops = 2 * (L - 1) * compose_cost(w)
compose_cost(w) = 2*w^3 + 2*w^2
```

解释：

- 近似仿射转移 `(A, c)` 的组合；
- `w=4`、`w=8` 下这个项会很快变重；
- 这正是本实验想确认的风险点。

### parallel depth

这里不用依赖图 level，而用一个更保守、更直接的 row-order 深度模型：

```text
baseline depth = n
region depth replacement = ceil(log2(L)) + 1
```

于是：

```text
estimated_parallel_depth
estimated_depth_reduction
```

衡量的是：

- 如果把一个长度 `L` 的顺序 region 换成 scan-style 处理，
- 在非常保守的串行深度模型下，最多能省下多少“行深度”。

### profit score

```text
estimated_profit_score =
    covered_row_ratio * estimated_depth_reduction
    - max(estimated_work_over_baseline - 1, 0)
```

这不是性能预测值，而是一个结构收益 vs 工作膨胀的保守裁决分数。

它为正，至少说明：

- 覆盖率和深度收益没有完全被额外工作吃掉。

它为负，则说明：

- 即使数学上可行，执行成本故事也站不住。

## support 判断规则

当前 summary 把单行结果判成 `support` 的条件是：

- `correctness_pass = true`
- `covered_row_ratio >= 0.10`
- `estimated_depth_reduction >= 0.20`
- `estimated_work_over_baseline <= 2.0`
- `estimated_profit_score > 0`

如果达不到这一组条件，但覆盖和深度还有一点点结构意义，就记为 `marginal`。

## 当前 formal_14 结果解读

当前这轮 study 的核心结论不是“算不对”，而是：

```text
算得对，但成本回不来。
```

更具体地说：

- 所有矩阵在 `region_seq / affine / scan_sim` 上都与 baseline 等价；
- 但没有任何矩阵在其最佳宽度上达到 `support`；
- `w=2` 的主要问题是覆盖率和深度收益普遍不够；
- `w=4` 的主要问题是虽然覆盖率上来了，但额外工作量也显著上升；
- `w=8` 几乎总能吃到接近全覆盖，但 compose / metadata 成本已经失控，不能作为主线依据。

组别上：

- SPD 组只有少量 `marginal`，没有 `support`；
- CFD 组结构上更像 region，但最佳宽度依然没有矩阵真正过线；
- 也就是说，CFD 比 SPD 更像“可能有点东西”，但还不到值得开 GPU prototype 的程度。

## 异常矩阵说明

### `ex11`

baseline 顺序解的 residual 本身较差。

这里的 `correctness_pass` 只能解释成：

- executor 与 baseline 等价；
- 不能解释成该矩阵本身数值上非常健康。

### `cfd2`

`x_ref` 与 `x_true` 的绝对误差较大，但 residual 很小。

这更像病态或尺度问题，不是 executor 路径自身错了。

## 当前方向裁决

当前结果支持的结论是：

```text
strict chain 不成立；
删边造 strict 长链不成立；
bounded-width region 在结构和数值等价上成立；
但 formal_14 的 CPU executor / cost study 仍不支持继续推进 scan-region 执行主线。
```

也就是说：

- 这条方向不是“数学假的”；
- 它更像是“数学上能做，但执行成本不够好”。

如果以后还要保留一点尾部空间，只适合写成：

- 一个很窄的 CFD / GMRES-ILU 观察窗口；
- 但不应作为当前主线继续投入 GPU executor 开发。

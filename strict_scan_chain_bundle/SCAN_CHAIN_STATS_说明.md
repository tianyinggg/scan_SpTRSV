# 严格单前驱链统计说明

本文档对应脚本 `analyze_scan_chains.py`。当前版本只实现最保守的“严格单前驱链”统计，不实现“前沿安全链”。

## 1. 输入定义

输入先读取原始稀疏矩阵 `A`，然后只做本项目需要的预处理：

- 只保留严格下三角部分
- 再补成单位对角
- 不做任何重排、level-set 置换或其他 permutation 处理

得到下三角稀疏矩阵 `L`。

对每一行 `i`，只看严格下三角依赖：

`pred[i] = { j | L[i, j] != 0 且 j < i }`

注意：

- 不包含对角元 `L[i, i]`
- 只统计 `j < i` 的前驱

实现位置：

- 读取矩阵：`load_matrix`
- 构建前驱：`build_predecessors`

## 2. 行分类规则

对每一行计算：

`pred_count[i] = len(pred[i])`

然后分成三类：

- `pred_count[i] == 0`：`absolute_head`
- `pred_count[i] == 1`：严格单前驱行
- `pred_count[i] > 1`：多前驱行，不能进入严格一元 scan 链

实现位置：

- `build_strict_parent`

## 3. 严格父节点定义

构造：

`strict_parent[i] = -1`

如果：

`pred_count[i] == 1`

则：

`strict_parent[i] = pred[i][0]`

否则：

`strict_parent[i] = -1`

这里没有“最深前驱选择”，也没有 level 概念。只要某行前驱数不是 1，就不能作为严格单前驱链中的内部节点。

## 4. 两类头部概念

当前版本显式区分两类“头部”概念：

### `absolute_head`

满足：

`pred_count[i] == 0`

这表示该行没有任何前驱，可以写成：

`x_i = beta_i`

### `strict_boundary_head`

严格单前驱链的链头边界不再要求 `pred_count[i] == 0`。

某一行 `head` 只要满足：

- 存在某个 `child`
- `pred_count[child] == 1`
- `pred[child][0] == head`

那么 `head` 就可以作为一条 strict single-predecessor chain 的起点边界。

这意味着：

- `head` 可以有 0 个前驱
- `head` 可以有 1 个前驱
- `head` 也可以有多个前驱

链头之前的依赖暂时视为链外边界输入。

真正需要严格满足“单前驱”的，是链头之后的节点。

如果某个 `head` 本身也是上游链的严格延续节点，并且它的父节点对它是唯一严格子节点，那么它不会被记为新的 boundary head，而是并入上游同一条线性链。

实现位置：

- `find_strict_boundary_heads`

## 5. 链提取规则

当前版本只提取最简单的线性链段。

规则如下：

1. 从 `strict_boundary_head` 出发提取链。
2. 对当前节点 `curr`，寻找满足下式的严格子节点：
   `pred_count[child] == 1` 且 `pred[child][0] == curr`
3. 如果这样的 `child` 恰好只有一个，则链继续。
4. 如果没有这样的 `child`，则链结束。
5. 如果有多个这样的 `child`，则最小版本先在 `curr` 处截断。

这里“严格子节点”指的就是 `strict_children[curr]`。

因此：

- 链头本身不一定是 strict single-predecessor row
- 链头之后的链中节点必须满足 `pred_count == 1`
- 分叉点本身只作为边界或截断点，不会继续穿过分叉

实现位置：

- `extract_strict_linear_chains`

## 6. 链长口径

链长按节点数统计，并包含链头本身。

例如：

- `0 -> 1`，链长为 2
- `0 -> 1 -> 2 -> 3`，链长为 4
- 若 `2` 作为新的 boundary head，则 `2 -> 3` 的链长为 2

当前版本只保留链长 `>= 2` 的链。只有单个孤立起点的长度为 1，不计入链统计。

## 7. CSV 输出字段

当前版本 CSV 默认采用：

首行英文列名 + 最后一行中文说明

例如：

- 首行：`matrix_name, n_rows, strict_chain_count, ...`
- 最后一行：`矩阵名, 行数, 严格链条数, ...`

这样做的目的是：

- 保留稳定的英文列名，便于脚本处理
- 在表格底部补一行中文含义，便于人工阅读

后续重新运行脚本时，会继续默认输出这种格式。

### `matrix_name`

矩阵文件名。

### `n_rows`

矩阵行数。

### `nnz`

矩阵总非零元数。

### `absolute_head_rows`

`pred_count == 0` 的行数。

### `absolute_head_row_ratio`

`absolute_head_rows / n_rows`

### `strict_boundary_head_rows`

严格单前驱链的边界链头数。这里的链头不要求自己是单前驱行，但要求它至少拥有一个严格单前驱子节点。

它包括例如：

- 无前驱的源节点，且后面接着严格单前驱链
- 有多个前驱的节点，且后面接着严格单前驱链
- 分叉截断之后重新开始的边界节点

### `strict_boundary_head_row_ratio`

`strict_boundary_head_rows / n_rows`

### `strict_single_pred_rows`

`pred_count == 1` 的行数。

### `strict_single_pred_row_ratio`

`strict_single_pred_rows / n_rows`

### `strict_multi_pred_rows`

`pred_count > 1` 的行数。

### `strict_multi_pred_row_ratio`

`strict_multi_pred_rows / n_rows`

### `strict_branch_rows`

严格分叉节点数，也就是严格子节点数大于 1 的节点数。

### `strict_branch_row_ratio`

`strict_branch_rows / n_rows`

### `strict_chain_count`

满足链长 `>= 2` 的严格线性链段数。

### `strict_chain_row_coverage`

所有严格链长度之和，也就是这些保守链段一共覆盖了多少个节点。

### `strict_chain_row_coverage_ratio`

`strict_chain_row_coverage / n_rows`

### `max_strict_chain_length`

最长严格链长度。

### `avg_strict_chain_length`

严格链平均长度。

### `strict_chain_len_ge_2`

长度 `>= 2` 的严格链条数。按当前口径，它与 `strict_chain_count` 相同。

### `strict_chain_len_ge_4`

长度 `>= 4` 的严格链条数。

### `strict_chain_len_ge_8`

长度 `>= 8` 的严格链条数。

### `strict_chain_len_ge_16`

长度 `>= 16` 的严格链条数。

### `strict_chain_len_ge_32`

长度 `>= 32` 的严格链条数。

### `strict_chain_len_ge_64`

长度 `>= 64` 的严格链条数。

## 8. 这版统计回答的问题

这版最小实现回答的是：

- 有多少行是完全无前驱的 `absolute_head`？
- 有多少行可以作为严格单前驱链的边界链头？
- 这些以边界输入为起点的严格单前驱链能有多长？
- 在哪里会因为严格分叉而截断？

## 9. 这版的优缺点

优点：

- 区分了“绝对链头”和“严格链边界链头”
- 仍然不依赖求解时序模拟
- 逻辑严格，容易解释

缺点：

- 统计仍然偏保守
- 分叉点本身不会穿过
- 多前驱但实际可 scan 的情况仍会被漏掉

后续如果要做第二层“前沿安全链”，可以在这个版本之上继续扩展。

## 10. 调试输出

脚本支持：

`--dump-chains`

开启后，会对每个矩阵额外打印一行 JSON，包含提取到的每条严格链。

如果希望同时落盘到文件，也可以使用：

`--dump-chains-out <path>`

输出格式为 JSONL，每行对应一个矩阵。

每条链包含：

- `head`：链头行号，从 0 开始
- `head_type`：`absolute_head` 或 `boundary_head`
- `head_pred_count`：链头自身前驱数
- `length`：链长
- `nodes`：链上节点序列，从链头到链尾

示例命令：

```bash
python analyze_scan_chains.py datasets --dump-chains
```

```bash
python analyze_scan_chains.py datasets --dump-chains --dump-chains-out strict_chain_dump.jsonl
```

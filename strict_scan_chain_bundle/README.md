# Strict Scan Chain Bundle

本目录打包了当前版本“SpTRSV 严格单前驱链统计”的完整交付内容。

## 文件说明

- `analyze_scan_chains.py`
  当前统计脚本。
- `SCAN_CHAIN_STATS_说明.md`
  指标定义、链头口径、链提取规则与调试输出说明。
- `scan_chain_stats_strict.csv`
  对 `datasets/` 目录全部 `.mtx` 文件跑出的统计结果。
- `strict_chain_dump.jsonl`
  每个矩阵一行 JSON，包含提取到的每条严格链节点序列。
- `run_stdout.txt`
  本次完整运行时的标准输出记录。

## 复现命令

```bash
python analyze_scan_chains.py datasets \
  --out strict_scan_chain_bundle/scan_chain_stats_strict.csv \
  --dump-chains \
  --dump-chains-out strict_scan_chain_bundle/strict_chain_dump.jsonl
```

## 当前口径

- 输入矩阵会先抽取严格下三角，再补单位对角
- 不引入任何来自 `2020-lu-sptrsv` 的重排或 permutation 逻辑
- 只统计严格下三角前驱 `j < i`
- 链头采用 `strict_boundary_head` 口径
- 链头本身不要求 `pred_count == 1`
- 链头之后的节点必须满足严格单前驱
- 碰到严格分叉时在当前节点截断，不穿过分叉

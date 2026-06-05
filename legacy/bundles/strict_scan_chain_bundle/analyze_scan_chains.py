#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse import coo_matrix, eye, tril


CSV_FIELDS = [
    "matrix_name",
    "n_rows",
    "nnz",
    "absolute_head_rows",
    "absolute_head_row_ratio",
    "strict_boundary_head_rows",
    "strict_boundary_head_row_ratio",
    "strict_single_pred_rows",
    "strict_single_pred_row_ratio",
    "strict_multi_pred_rows",
    "strict_multi_pred_row_ratio",
    "strict_branch_rows",
    "strict_branch_row_ratio",
    "strict_chain_count",
    "strict_chain_row_coverage",
    "strict_chain_row_coverage_ratio",
    "max_strict_chain_length",
    "avg_strict_chain_length",
    "strict_chain_len_ge_2",
    "strict_chain_len_ge_4",
    "strict_chain_len_ge_8",
    "strict_chain_len_ge_16",
    "strict_chain_len_ge_32",
    "strict_chain_len_ge_64",
]

CSV_FIELD_LABELS = {
    "matrix_name": "矩阵名",
    "n_rows": "行数",
    "nnz": "非零元数",
    "absolute_head_rows": "绝对链头行数",
    "absolute_head_row_ratio": "绝对链头行占比",
    "strict_boundary_head_rows": "严格边界链头行数",
    "strict_boundary_head_row_ratio": "严格边界链头行占比",
    "strict_single_pred_rows": "严格单前驱行数",
    "strict_single_pred_row_ratio": "严格单前驱行占比",
    "strict_multi_pred_rows": "多前驱行数",
    "strict_multi_pred_row_ratio": "多前驱行占比",
    "strict_branch_rows": "严格分叉节点数",
    "strict_branch_row_ratio": "严格分叉节点占比",
    "strict_chain_count": "严格链条数",
    "strict_chain_row_coverage": "严格链覆盖节点数",
    "strict_chain_row_coverage_ratio": "严格链覆盖占比",
    "max_strict_chain_length": "最长严格链长度",
    "avg_strict_chain_length": "平均严格链长度",
    "strict_chain_len_ge_2": "长度至少2的严格链条数",
    "strict_chain_len_ge_4": "长度至少4的严格链条数",
    "strict_chain_len_ge_8": "长度至少8的严格链条数",
    "strict_chain_len_ge_16": "长度至少16的严格链条数",
    "strict_chain_len_ge_32": "长度至少32的严格链条数",
    "strict_chain_len_ge_64": "长度至少64的严格链条数",
}


def load_matrix(path: Path, triangle: str = "lower"):
    """
    Read a Matrix Market file and apply only the local SpTRSV preprocessing:
    keep only the strict lower triangle and then add a unit diagonal.
    No reordering or permutation logic is introduced here.
    """
    mat = mmread(str(path))
    if not hasattr(mat, "tocsr"):
        mat = coo_matrix(mat)

    a_csr = mat.tocsr()
    if a_csr.shape[0] != a_csr.shape[1]:
        raise ValueError(f"Matrix is not square: {path.name}, shape={a_csr.shape}")

    if triangle != "lower":
        raise ValueError(f"Unsupported triangle type: {triangle}")

    a_csr.sum_duplicates()
    tri_csr = tril(a_csr, k=-1, format="csr")
    tri_csr = tri_csr + eye(a_csr.shape[0], format="csr", dtype=tri_csr.dtype)
    tri_csr.sum_duplicates()
    tri_csr.sort_indices()
    tri_csr.eliminate_zeros()
    return tri_csr


def build_predecessors(a_csr):
    """Build pred[i] = {j | A[i, j] != 0 and j < i}."""
    n_rows = a_csr.shape[0]
    pred = [[] for _ in range(n_rows)]

    indptr = a_csr.indptr
    indices = a_csr.indices

    for i in range(n_rows):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        row_cols = indices[row_start:row_end]
        pred[i] = [int(j) for j in row_cols if j < i]

    return pred


def build_strict_parent(pred):
    """Keep only exact single-predecessor rows as strict-parent edges."""
    n = len(pred)
    pred_count = np.zeros(n, dtype=np.int64)
    strict_parent = np.full(n, -1, dtype=np.int64)

    for i, row_pred in enumerate(pred):
        count = len(row_pred)
        pred_count[i] = count
        if count == 1:
            strict_parent[i] = row_pred[0]

    return pred_count, strict_parent


def build_strict_children(strict_parent):
    """Build reverse adjacency from strict_parent."""
    n = len(strict_parent)
    strict_children = [[] for _ in range(n)]

    for child, parent in enumerate(strict_parent):
        if parent != -1:
            strict_children[int(parent)].append(child)

    return strict_children


def find_strict_boundary_heads(pred_count, strict_parent, strict_children):
    """
    Find boundary heads of strict single-predecessor chains.

    A row may be a boundary head regardless of its own predecessor count.
    It qualifies if:
    - it has at least one strict child, and
    - it is not itself a strict in-chain continuation of its parent
    """
    n = len(pred_count)
    boundary_heads = np.zeros(n, dtype=bool)

    for i in range(n):
        if len(strict_children[i]) == 0:
            continue

        if pred_count[i] != 1:
            boundary_heads[i] = True
            continue

        parent = int(strict_parent[i])
        if parent == -1:
            boundary_heads[i] = True
            continue

        if len(strict_children[parent]) != 1:
            boundary_heads[i] = True
            continue

    return boundary_heads


def extract_strict_linear_chains(strict_children, boundary_heads):
    """
    Extract conservative strict single-predecessor chains.

    Rules:
    - Boundary heads are given by boundary_heads.
    - A chain continues only when the current node has exactly one strict child.
    - If a node has zero or multiple strict children, the chain stops there.
    - Only keep chains with length >= 2.
    """
    chain_lengths = []

    for head in range(len(boundary_heads)):
        if not boundary_heads[head]:
            continue

        length = 1
        current = head

        while True:
            children = strict_children[current]
            if len(children) != 1:
                break
            current = children[0]
            length += 1

        if length >= 2:
            chain_lengths.append(length)

    return chain_lengths


def extract_strict_linear_chain_nodes(strict_children, boundary_heads):
    """Extract node sequences for each strict chain with length >= 2."""
    chains = []

    for head in range(len(boundary_heads)):
        if not boundary_heads[head]:
            continue

        chain = [head]
        current = head

        while True:
            children = strict_children[current]
            if len(children) != 1:
                break
            current = children[0]
            chain.append(current)

        if len(chain) >= 2:
            chains.append(chain)

    return chains


def analyze_matrix(path: Path, triangle="lower", dump_chains=False):
    a = load_matrix(path, triangle=triangle)

    n_rows = int(a.shape[0])
    nnz = int(a.nnz)

    pred = build_predecessors(a)
    pred_count, strict_parent = build_strict_parent(pred)
    strict_children = build_strict_children(strict_parent)
    boundary_heads = find_strict_boundary_heads(pred_count, strict_parent, strict_children)

    absolute_head_rows = int(np.count_nonzero(pred_count == 0))
    strict_boundary_head_rows = int(np.count_nonzero(boundary_heads))
    strict_single_pred_rows = int(np.count_nonzero(pred_count == 1))
    strict_multi_pred_rows = int(np.count_nonzero(pred_count > 1))
    strict_branch_rows = int(sum(1 for children in strict_children if len(children) > 1))

    chain_nodes = extract_strict_linear_chain_nodes(strict_children, boundary_heads)
    chain_lengths = [len(chain) for chain in chain_nodes]
    strict_chain_count = len(chain_lengths)
    strict_chain_row_coverage = int(sum(chain_lengths))
    max_strict_chain_length = max(chain_lengths) if chain_lengths else 0
    avg_strict_chain_length = float(np.mean(chain_lengths)) if chain_lengths else 0.0

    result = {
        "matrix_name": path.name,
        "n_rows": n_rows,
        "nnz": nnz,
        "absolute_head_rows": absolute_head_rows,
        "absolute_head_row_ratio": (absolute_head_rows / n_rows) if n_rows > 0 else 0.0,
        "strict_boundary_head_rows": strict_boundary_head_rows,
        "strict_boundary_head_row_ratio": (strict_boundary_head_rows / n_rows) if n_rows > 0 else 0.0,
        "strict_single_pred_rows": strict_single_pred_rows,
        "strict_single_pred_row_ratio": (strict_single_pred_rows / n_rows) if n_rows > 0 else 0.0,
        "strict_multi_pred_rows": strict_multi_pred_rows,
        "strict_multi_pred_row_ratio": (strict_multi_pred_rows / n_rows) if n_rows > 0 else 0.0,
        "strict_branch_rows": strict_branch_rows,
        "strict_branch_row_ratio": (strict_branch_rows / n_rows) if n_rows > 0 else 0.0,
        "strict_chain_count": strict_chain_count,
        "strict_chain_row_coverage": strict_chain_row_coverage,
        "strict_chain_row_coverage_ratio": (strict_chain_row_coverage / n_rows) if n_rows > 0 else 0.0,
        "max_strict_chain_length": int(max_strict_chain_length),
        "avg_strict_chain_length": avg_strict_chain_length,
        "strict_chain_len_ge_2": int(sum(1 for x in chain_lengths if x >= 2)),
        "strict_chain_len_ge_4": int(sum(1 for x in chain_lengths if x >= 4)),
        "strict_chain_len_ge_8": int(sum(1 for x in chain_lengths if x >= 8)),
        "strict_chain_len_ge_16": int(sum(1 for x in chain_lengths if x >= 16)),
        "strict_chain_len_ge_32": int(sum(1 for x in chain_lengths if x >= 32)),
        "strict_chain_len_ge_64": int(sum(1 for x in chain_lengths if x >= 64)),
    }

    if dump_chains:
        dumped = []
        for chain in chain_nodes:
            head = chain[0]
            dumped.append(
                {
                    "head": head,
                    "head_type": "absolute_head" if pred_count[head] == 0 else "boundary_head",
                    "head_pred_count": int(pred_count[head]),
                    "length": len(chain),
                    "nodes": chain,
                }
            )
        result["strict_chains"] = dumped

    return result


def collect_mtx_files(input_path: Path):
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.mtx"))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze strict single-predecessor chains for SpTRSV matrices."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="datasets",
        help="Input .mtx file or directory containing .mtx files (default: datasets)",
    )
    parser.add_argument(
        "--out",
        default="scan_chain_stats_strict.csv",
        help="Output CSV path (default: scan_chain_stats_strict.csv)",
    )
    parser.add_argument(
        "--dump-chains",
        action="store_true",
        help="Print each extracted strict chain as JSON lines",
    )
    parser.add_argument(
        "--dump-chains-out",
        default=None,
        help="Optional JSONL output path for dumped strict chains",
    )
    parser.add_argument(
        "--triangle",
        default="lower",
        choices=["lower"],
        help="Triangle type after preprocessing (currently only lower is supported)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.out)
    dump_chains_out_path = Path(args.dump_chains_out) if args.dump_chains_out else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    mtx_files = collect_mtx_files(input_path)
    if not mtx_files:
        print(f"No .mtx files found in {input_path}")

    results = []
    dump_lines = []
    for mtx in mtx_files:
        try:
            res = analyze_matrix(mtx, triangle=args.triangle, dump_chains=args.dump_chains)
            results.append(res)
            if args.dump_chains:
                chain_dump = {
                    "matrix_name": mtx.name,
                    "strict_chains": res.get("strict_chains", []),
                }
                chain_dump_line = json.dumps(chain_dump, ensure_ascii=False)
                print(chain_dump_line)
                dump_lines.append(chain_dump_line)
        except Exception as exc:
            print(f"[WARN] Skip {mtx.name}: {exc}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row[field] for field in CSV_FIELDS})
        writer.writerow({field: CSV_FIELD_LABELS[field] for field in CSV_FIELDS})

    if dump_chains_out_path is not None:
        dump_chains_out_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_chains_out_path.open("w", encoding="utf-8") as f:
            for line in dump_lines:
                f.write(line + "\n")

    print(f"Done. Processed {len(results)} matrices. Output: {output_path}")


if __name__ == "__main__":
    main()

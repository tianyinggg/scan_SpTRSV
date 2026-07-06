#!/usr/bin/env python3
"""Download selected SuiteSparse matrices and generate SPCG-style L factors."""

from __future__ import annotations

import argparse
import csv
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_MATRIX_DIR = DATA_DIR / "matrices" / "real"
DEFAULT_FACTOR_DIR = DATA_DIR / "factors" / "spcg"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "results"
    / "factor_preparation"
    / "suitesparse_spd_structural_thermal_manifest.csv"
)
DEFAULT_CFD_MANIFEST = (
    REPO_ROOT
    / "results"
    / "factor_preparation"
    / "suitesparse_cfd_nonsym_pde_manifest.csv"
)
DEFAULT_ALL_MANIFEST = (
    REPO_ROOT
    / "results"
    / "factor_preparation"
    / "suitesparse_factor_manifest.csv"
)
SUITESPARSE_MM_BASE_URL = "https://sparse.tamu.edu/MM"
USER_AGENT = "scan-SpTRSV/0.1"


@dataclass(frozen=True)
class SuiteSparseCandidate:
    group: str
    name: str
    problem_class: str
    scenario: str
    notes: str

    @property
    def url(self) -> str:
        return f"{SUITESPARSE_MM_BASE_URL}/{self.group}/{self.name}.tar.gz"


@dataclass(frozen=True)
class MatrixMarketInfo:
    n_rows: int
    n_cols: int
    reported_nnz: int
    field: str
    symmetry: str


MANIFEST_FIELDS = [
    "matrix_name",
    "suitesparse_group",
    "suitesparse_url",
    "problem_class",
    "scenario",
    "notes",
    "matrix_path",
    "factor_l_path",
    "factor_u_path",
    "archive_bytes",
    "n_rows",
    "n_cols",
    "reported_nnz",
    "field",
    "symmetry",
    "download_status",
    "factor_status",
    "factor_method",
    "fill_factor",
    "drop_tol",
    "diagonal_shift",
    "permc_spec",
    "diag_pivot_thresh",
    "elapsed_seconds",
    "error",
]


def spd_structural_thermal_candidates() -> list[SuiteSparseCandidate]:
    """Curated SPD / structural / thermal candidates for PCG/IC/ILU tests."""
    return [
        SuiteSparseCandidate(
            "Schmid",
            "thermal1",
            "thermal-spd",
            "PCG/IC/ILU",
            "Thermal problem; useful non-structural SPD contrast.",
        ),
        SuiteSparseCandidate(
            "GHS_psdef",
            "apache1",
            "spd-fem",
            "PCG/IC/ILU",
            "Positive-definite FEM benchmark; moderate size.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk06",
            "structural-spd",
            "PCG/IC/ILU",
            "BCS structural stiffness matrix.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk07",
            "structural-spd",
            "PCG/IC/ILU",
            "BCS structural stiffness matrix.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk08",
            "structural-spd",
            "PCG/IC/ILU",
            "BCS structural stiffness matrix.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk09",
            "structural-spd",
            "PCG/IC/ILU",
            "BCS structural stiffness matrix.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk10",
            "structural-spd",
            "PCG/IC/ILU",
            "BCS structural stiffness matrix.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk11",
            "structural-spd",
            "PCG/IC/ILU",
            "BCS structural stiffness matrix.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk12",
            "structural-spd",
            "PCG/IC/ILU",
            "BCS structural stiffness matrix.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk13",
            "structural-spd",
            "PCG/IC/ILU",
            "BCS structural stiffness matrix.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk14",
            "structural-spd",
            "PCG/IC/ILU",
            "BCS structural stiffness matrix.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk15",
            "structural-spd",
            "PCG/IC/ILU",
            "BCS structural stiffness matrix.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk16",
            "structural-spd",
            "PCG/IC/ILU",
            "BCS structural stiffness matrix.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk17",
            "structural-spd",
            "PCG/IC/ILU",
            "BCS structural stiffness matrix.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk18",
            "structural-spd",
            "PCG/IC/ILU",
            "BCS structural stiffness matrix.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk03",
            "structural-spd-small",
            "PCG/IC/ILU",
            "Small BCS structural sanity check.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk04",
            "structural-spd-small",
            "PCG/IC/ILU",
            "Small BCS structural sanity check.",
        ),
        SuiteSparseCandidate(
            "HB",
            "bcsstk05",
            "structural-spd-small",
            "PCG/IC/ILU",
            "Small BCS structural sanity check.",
        ),
        SuiteSparseCandidate(
            "Schmid",
            "thermal2",
            "thermal-spd",
            "PCG/IC/ILU",
            "Larger thermal problem; run after the smaller set is stable.",
        ),
        SuiteSparseCandidate(
            "GHS_psdef",
            "apache2",
            "spd-fem",
            "PCG/IC/ILU",
            "Larger positive-definite FEM benchmark.",
        ),
    ]


def cfd_nonsym_pde_candidates() -> list[SuiteSparseCandidate]:
    """Curated CFD / nonsymmetric PDE candidates for GMRES/ILU tests."""
    return [
        SuiteSparseCandidate(
            "Rothberg",
            "cfd1",
            "cfd-nonsym-pde",
            "GMRES/ILU",
            "CFD benchmark; nonsymmetric PDE-style ILU target.",
        ),
        SuiteSparseCandidate(
            "Rothberg",
            "cfd2",
            "cfd-nonsym-pde",
            "GMRES/ILU",
            "CFD benchmark; larger companion to cfd1.",
        ),
        SuiteSparseCandidate(
            "FIDAP",
            "ex11",
            "fem-fluid-nonsym-pde",
            "GMRES/ILU",
            "FIDAP finite-element fluid/PDE matrix; GMRES/ILU contrast.",
        ),
        SuiteSparseCandidate(
            "FIDAP",
            "ex19",
            "fem-fluid-nonsym-pde",
            "GMRES/ILU",
            "FIDAP finite-element fluid/PDE matrix; moderate size.",
        ),
        SuiteSparseCandidate(
            "FIDAP",
            "ex15",
            "fem-fluid-nonsym-pde",
            "GMRES/ILU",
            "Smaller FIDAP nonsymmetric PDE sanity case.",
        ),
        SuiteSparseCandidate(
            "Simon",
            "raefsky3",
            "cfd-nonsym-pde",
            "GMRES/ILU",
            "RaeFSKY CFD/PDE matrix; useful larger nonsymmetric case.",
        ),
    ]


def candidates_for_set(candidate_set: str) -> list[SuiteSparseCandidate]:
    spd = spd_structural_thermal_candidates()
    cfd = cfd_nonsym_pde_candidates()
    if candidate_set == "spd":
        return spd
    if candidate_set == "cfd":
        return cfd
    if candidate_set == "all":
        return spd + cfd
    raise ValueError(f"Unknown candidate set: {candidate_set}")


def default_manifest_for_set(candidate_set: str) -> Path:
    if candidate_set == "spd":
        return DEFAULT_MANIFEST
    if candidate_set == "cfd":
        return DEFAULT_CFD_MANIFEST
    if candidate_set == "all":
        return DEFAULT_ALL_MANIFEST
    raise ValueError(f"Unknown candidate set: {candidate_set}")


def resolve_repo_path(path_arg: str | None, default: Path) -> Path:
    if path_arg is None:
        return default
    path = Path(path_arg).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def float_tag(value: float | None) -> str:
    if value is None:
        return "default"
    return f"{value:g}"


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"none", "default", ""}:
        return None
    return float(value)


def expected_factor_paths(
    matrix_path: Path,
    factor_dir: Path,
    method: str,
    fill_factor: float,
    drop_tol: float | None,
    diagonal_shift: float,
    permc_spec: str = "NATURAL",
) -> tuple[Path, Path]:
    shift_tag = "" if diagonal_shift == 0 else f"_shift{diagonal_shift:g}"
    permc_tag = "" if permc_spec == "NATURAL" else f"_permc{permc_spec.lower()}"
    base_name = (
        f"spcg_{method}_l_{matrix_path.stem}"
        f"_fill{fill_factor:g}_drop{float_tag(drop_tol)}{shift_tag}"
        f"{permc_tag}_sp0"
    )
    matrix_factor_dir = factor_dir / matrix_path.stem
    return (
        matrix_factor_dir / f"{base_name}.mtx",
        matrix_factor_dir / f"{base_name.replace('_l_', '_u_', 1)}.mtx",
    )


def fetch_remote_size(url: str, timeout: int) -> int | None:
    request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            return int(content_length) if content_length else None
    except (HTTPError, URLError, TimeoutError):
        return None


def download_archive(url: str, timeout: int) -> tuple[Path, int]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz")
    tmp_path = Path(tmp.name)
    total = 0
    try:
        with tmp:
            with urlopen(request, timeout=timeout) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    tmp.write(chunk)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path, total


def extract_matrix_market(
    archive_path: Path,
    candidate: SuiteSparseCandidate,
    matrix_dir: Path,
    overwrite: bool,
) -> tuple[Path, str]:
    dest_dir = matrix_dir / candidate.name
    dest_path = dest_dir / f"{candidate.name}.mtx"
    if dest_path.exists() and not overwrite:
        return dest_path, "existing"

    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and Path(member.name).suffix == ".mtx"
        ]
        if not members:
            raise ValueError(f"No .mtx file found in {archive_path.name}")

        preferred = [
            member
            for member in members
            if Path(member.name).name == f"{candidate.name}.mtx"
        ]
        member = preferred[0] if preferred else members[0]
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"Cannot extract {member.name} from {archive_path.name}")
        with source, dest_path.open("wb") as dest:
            shutil.copyfileobj(source, dest)

    return dest_path, "downloaded"


def read_matrix_market_info(path: Path) -> MatrixMarketInfo:
    with path.open("r", encoding="utf-8", errors="replace") as file_obj:
        banner = file_obj.readline().strip().split()
        if len(banner) != 5 or banner[0] != "%%MatrixMarket":
            raise ValueError(f"Invalid Matrix Market banner: {path}")
        field = banner[3].lower()
        symmetry = banner[4].lower()

        for line in file_obj:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid Matrix Market size line: {path}")
            return MatrixMarketInfo(
                n_rows=int(parts[0]),
                n_cols=int(parts[1]),
                reported_nnz=int(parts[2]),
                field=field,
                symmetry=symmetry,
            )

    raise ValueError(f"Missing Matrix Market size line: {path}")


def generate_spcg_factors(
    matrix_path: Path,
    factor_dir: Path,
    method: str,
    fill_factor: float,
    drop_tol: float | None,
    diagonal_shift: float,
    permc_spec: str,
    diag_pivot_thresh: float,
) -> tuple[Path, Path]:
    import scipy.io as scipy_io
    import scipy.sparse as scipy_sparse
    import scipy.sparse.linalg as scipy_linalg

    matrix = scipy_io.mmread(str(matrix_path))
    if not hasattr(matrix, "tocsc"):
        matrix = scipy_sparse.coo_matrix(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix is not square: {matrix_path.name}, shape={matrix.shape}")

    matrix = matrix.tocsc()
    if diagonal_shift != 0:
        identity = scipy_sparse.eye(
            matrix.shape[0], format="csc", dtype=matrix.dtype
        )
        matrix = matrix + diagonal_shift * identity

    if method == "splu":
        factors = scipy_linalg.splu(
            matrix,
            permc_spec=permc_spec,
            diag_pivot_thresh=diag_pivot_thresh,
        )
    elif method == "spilu":
        kwargs = {
            "fill_factor": fill_factor,
            "permc_spec": permc_spec,
            "diag_pivot_thresh": diag_pivot_thresh,
        }
        if drop_tol is not None:
            kwargs["drop_tol"] = drop_tol
        factors = scipy_linalg.spilu(matrix, **kwargs)
    else:
        raise ValueError(f"Unsupported factor method: {method}")

    l_path, u_path = expected_factor_paths(
        matrix_path=matrix_path,
        factor_dir=factor_dir,
        method=method,
        fill_factor=fill_factor,
        drop_tol=drop_tol,
        diagonal_shift=diagonal_shift,
        permc_spec=permc_spec,
    )
    l_path.parent.mkdir(parents=True, exist_ok=True)
    scipy_io.mmwrite(str(l_path), factors.L)
    scipy_io.mmwrite(str(u_path), factors.U)
    return l_path, u_path


def select_candidates(
    candidates: list[SuiteSparseCandidate],
    only: str | None,
    limit: int | None,
) -> list[SuiteSparseCandidate]:
    selected = candidates
    if only:
        requested = [name.strip() for name in only.split(",") if name.strip()]
        by_name = {candidate.name: candidate for candidate in candidates}
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise ValueError(f"Unknown candidate(s): {', '.join(missing)}")
        selected = [by_name[name] for name in requested]
    if limit is not None:
        selected = selected[:limit]
    return selected


def blank_row(candidate: SuiteSparseCandidate, args) -> dict[str, str]:
    return {
        "matrix_name": candidate.name,
        "suitesparse_group": candidate.group,
        "suitesparse_url": candidate.url,
        "problem_class": candidate.problem_class,
        "scenario": candidate.scenario,
        "notes": candidate.notes,
        "matrix_path": "",
        "factor_l_path": "",
        "factor_u_path": "",
        "archive_bytes": "",
        "n_rows": "",
        "n_cols": "",
        "reported_nnz": "",
        "field": "",
        "symmetry": "",
        "download_status": "",
        "factor_status": "",
        "factor_method": args.factor_method,
        "fill_factor": f"{args.fill_factor:g}",
        "drop_tol": float_tag(args.drop_tol),
        "diagonal_shift": f"{args.diagonal_shift:g}",
        "permc_spec": args.permc_spec,
        "diag_pivot_thresh": f"{args.diag_pivot_thresh:g}",
        "elapsed_seconds": "",
        "error": "",
    }


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def prepare_candidate(candidate: SuiteSparseCandidate, args) -> dict[str, str]:
    started = time.time()
    row = blank_row(candidate, args)
    archive_path: Path | None = None
    matrix_path = args.matrix_dir / candidate.name / f"{candidate.name}.mtx"

    try:
        remote_size = fetch_remote_size(candidate.url, args.timeout)
        if remote_size is not None:
            row["archive_bytes"] = str(remote_size)
            if args.max_download_mb is not None:
                max_bytes = int(args.max_download_mb * 1024 * 1024)
                if remote_size > max_bytes and not matrix_path.exists():
                    row["download_status"] = "skipped_too_large"
                    row["factor_status"] = "skipped_no_matrix"
                    row["error"] = (
                        f"archive {remote_size} bytes exceeds "
                        f"--max-download-mb {args.max_download_mb:g}"
                    )
                    return row

        if matrix_path.exists() and not args.overwrite_matrices:
            row["download_status"] = "existing"
        else:
            archive_path, archive_bytes = download_archive(candidate.url, args.timeout)
            row["archive_bytes"] = str(archive_bytes)
            matrix_path, download_status = extract_matrix_market(
                archive_path,
                candidate,
                args.matrix_dir,
                overwrite=args.overwrite_matrices,
            )
            row["download_status"] = download_status

        row["matrix_path"] = str(matrix_path.relative_to(REPO_ROOT))
        info = read_matrix_market_info(matrix_path)
        row["n_rows"] = str(info.n_rows)
        row["n_cols"] = str(info.n_cols)
        row["reported_nnz"] = str(info.reported_nnz)
        row["field"] = info.field
        row["symmetry"] = info.symmetry

        if args.download_only:
            row["factor_status"] = "skipped_download_only"
            return row

        l_path, u_path = expected_factor_paths(
            matrix_path=matrix_path,
            factor_dir=args.factor_dir,
            method=args.factor_method,
            fill_factor=args.fill_factor,
            drop_tol=args.drop_tol,
            diagonal_shift=args.diagonal_shift,
            permc_spec=args.permc_spec,
        )
        row["factor_l_path"] = str(l_path.relative_to(REPO_ROOT))
        row["factor_u_path"] = str(u_path.relative_to(REPO_ROOT))

        if l_path.exists() and u_path.exists() and not args.overwrite_factors:
            row["factor_status"] = "existing"
            return row

        l_path, u_path = generate_spcg_factors(
            matrix_path=matrix_path,
            factor_dir=args.factor_dir,
            method=args.factor_method,
            fill_factor=args.fill_factor,
            drop_tol=args.drop_tol,
            diagonal_shift=args.diagonal_shift,
            permc_spec=args.permc_spec,
            diag_pivot_thresh=args.diag_pivot_thresh,
        )
        row["factor_l_path"] = str(l_path.relative_to(REPO_ROOT))
        row["factor_u_path"] = str(u_path.relative_to(REPO_ROOT))
        row["factor_status"] = "generated"
    except Exception as exc:
        if not row["download_status"]:
            row["download_status"] = "failed"
        if not row["factor_status"]:
            row["factor_status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)
        row["elapsed_seconds"] = f"{time.time() - started:.3f}"

    return row


def print_candidate_table(candidates: list[SuiteSparseCandidate]) -> None:
    print("name,group,problem_class,scenario,url")
    for candidate in candidates:
        print(
            f"{candidate.name},{candidate.group},{candidate.problem_class},"
            f"{candidate.scenario},{candidate.url}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download a curated SuiteSparse SPD/structural/thermal matrix set "
            "and generate SPCG-style SuperLU L factors."
        )
    )
    parser.add_argument(
        "--candidate-set",
        choices=["spd", "cfd", "all"],
        default="spd",
        help=(
            "Curated SuiteSparse candidate set: spd for SPD/structural/thermal, "
            "cfd for CFD/nonsymmetric PDE, all for both (default: spd)"
        ),
    )
    parser.add_argument(
        "--matrix-dir",
        default=None,
        help="Output directory for downloaded matrices (default: data/matrices/real)",
    )
    parser.add_argument(
        "--factor-dir",
        default=None,
        help="Output directory for generated factors (default: data/factors/spcg)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "CSV manifest output "
            "(default: results/factor_preparation/suitesparse_spd_structural_thermal_manifest.csv)"
        ),
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated candidate names to process, e.g. thermal1,bcsstk13",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N selected candidates",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List curated candidates and exit",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download/extract matrices without factorization",
    )
    parser.add_argument(
        "--overwrite-matrices",
        action="store_true",
        help="Re-download and replace existing extracted .mtx files",
    )
    parser.add_argument(
        "--overwrite-factors",
        action="store_true",
        help="Regenerate L/U factors even if matching files already exist",
    )
    parser.add_argument(
        "--factor-method",
        choices=["spilu", "splu"],
        default="spilu",
        help="SuperLU factorization method (default: spilu)",
    )
    parser.add_argument(
        "--fill-factor",
        type=float,
        default=10.0,
        help="SuperLU fill_factor for spilu (default: 10)",
    )
    parser.add_argument(
        "--drop-tol",
        default="1e-4",
        help="SuperLU drop_tol for spilu; use 'default' to omit it (default: 1e-4)",
    )
    parser.add_argument(
        "--diagonal-shift",
        type=float,
        default=0.0,
        help="Optional diagonal shift before factorization (default: 0)",
    )
    parser.add_argument(
        "--permc-spec",
        default="NATURAL",
        choices=["NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"],
        help="SuperLU column permutation strategy (default: NATURAL)",
    )
    parser.add_argument(
        "--diag-pivot-thresh",
        type=float,
        default=1e-15,
        help="SuperLU diagonal pivot threshold (default: 1e-15)",
    )
    parser.add_argument(
        "--max-download-mb",
        type=float,
        default=None,
        help="Skip missing matrices whose archive exceeds this size",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Network timeout in seconds per request (default: 120)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.drop_tol = parse_optional_float(args.drop_tol)
    args.matrix_dir = resolve_repo_path(args.matrix_dir, DEFAULT_MATRIX_DIR)
    args.factor_dir = resolve_repo_path(args.factor_dir, DEFAULT_FACTOR_DIR)
    args.manifest = resolve_repo_path(
        args.manifest, default_manifest_for_set(args.candidate_set)
    )

    candidates = select_candidates(
        candidates_for_set(args.candidate_set), args.only, args.limit
    )
    if args.list:
        print_candidate_table(candidates)
        return

    rows = []
    for candidate in candidates:
        print(f"[prepare] {candidate.group}/{candidate.name}", flush=True)
        rows.append(prepare_candidate(candidate, args))
        write_manifest(rows, args.manifest)

    generated = sum(1 for row in rows if row["factor_status"] == "generated")
    existing = sum(1 for row in rows if row["factor_status"] == "existing")
    failed = sum(1 for row in rows if row["factor_status"] == "failed")
    print(
        "Completed: "
        f"{generated} generated, {existing} existing, {failed} failed. "
        f"Manifest: {args.manifest.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()

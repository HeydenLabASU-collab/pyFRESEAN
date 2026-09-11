#!/usr/bin/env python3
"""Benchmark pyfresean vs FRESEAN COARSE (C) on HEWL.

Modes (use separately for clean sweeps):
  cg      — pyfresean coarse-grain once; writes cg_reference/pyfresean/
  fresean — spectral only (loads cg_reference/pyfresean to node-local tmp)
  c       — C covar+eigen only (workdir on node-local tmp)
  all     — cg + fresean + c in one process (legacy convenience)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import MDAnalysis as mda

from pyfresean import Align, FRESEAN
from pyfresean.benchmark_keys import (
    BENCH_T_C_COVAR,
    BENCH_T_C_EIGEN,
    BENCH_T_C_TOTAL,
    BENCH_T_PY_COARSE,
    BENCH_T_PY_FRESEAN,
    BENCH_T_PY_TOTAL,
    BENCH_T_SPECTRAL,
    BENCH_T_TOTAL,
)
from pyfresean.coarsegrain.coarse import CoarseGrain
from pyfresean.tests.c_ref.ala_dipeptide_gas_300K import N_CORR, SIGMA_CM1
from pyfresean.tests.c_ref.hewl_solution_303K import (
    DT_PS,
    N_FRAMES_DEFAULT,
    resolve_hewl_solution_303K_paths,
)

BENCH_DIR = Path(__file__).resolve().parent
PYFRESEAN_ROOT = BENCH_DIR.parent.parent
DEFAULT_CG_REFERENCE_ROOT = BENCH_DIR / "cg_reference"
DEFAULT_C_INPUTS = DEFAULT_CG_REFERENCE_ROOT / "c_ref"
DEFAULT_PY_CG_INPUTS = DEFAULT_CG_REFERENCE_ROOT / "pyfresean"
DEFAULT_RESULTS = BENCH_DIR / "results"
DEFAULT_CG_REFERENCE = DEFAULT_PY_CG_INPUTS
DEFAULT_CG_CACHE = DEFAULT_PY_CG_INPUTS
DEFAULT_C_SPECTRAL = DEFAULT_RESULTS / "c_spectral"
DEFAULT_PY_CASES = DEFAULT_RESULTS / "py_cases"
DEFAULT_FRESEAN_BIN = os.environ.get("FRESEAN_BIN", "fresean")
WIN_SIGMA = 10.0
CPU_COUNTS = (1, 2, 4, 8, 16, 32, 48)

CaseResolver = Callable[[int], tuple[int, int, int]]

BENCH_CASES: dict[str, CaseResolver] = {
    "omp1_njobs_n_nworkers_1": lambda n: (1, n, 1),
    "omp1_njobs_n_nworkers_2": lambda n: (1, n, min(2, n)),
    "omp1_njobs_n_nworkers_n": lambda n: (1, n, n),
    "omp_n_njobs_1": lambda n: (n, 1, 1),
}


@dataclass
class BenchmarkResult:
    mode: str
    case: str
    ncpus: int
    omp_threads: int
    py_n_jobs: int
    py_n_workers: int
    hostname: str
    slurm_job_id: str
    timestamp_utc: str
    bench_t_py_coarse: float
    bench_t_py_fresean: float
    bench_t_py_total: float
    bench_t_c_covar: float
    bench_t_c_eigen: float
    bench_t_c_total: float
    py_coarse: dict[str, float] | None
    py_fresean: dict[str, float] | None
    n_frames: int
    n_corr: int
    dt_ps: float


def _tmp_root() -> Path:
    root = Path(os.environ.get("SLURM_TMPDIR", os.environ.get("TMPDIR", "/tmp")))
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_case(case: str, ncpus: int) -> tuple[int, int, int]:
    if case not in BENCH_CASES:
        raise ValueError(f"unknown case {case!r}; choose: {', '.join(BENCH_CASES)}")
    return BENCH_CASES[case](ncpus)


def resolve_parallel_config(config: str, ncpus: int) -> tuple[str, int, int, int]:
    """Legacy alias mapping for older submit scripts."""
    aliases = {
        "a": "omp1_njobs_n_nworkers_1",
        "b": "omp_n_njobs_1",
        "c": "omp1_njobs_n_nworkers_2",
        "omp1_njobs_n": "omp1_njobs_n_nworkers_1",
        "omp_half_njobs_2": "omp1_njobs_n_nworkers_2",
    }
    case = aliases.get(config.lower(), config)
    omp_threads, n_jobs, n_workers = resolve_case(case, ncpus)
    return case, omp_threads, n_jobs, n_workers


def _set_py_thread_env(omp_threads: int = 1) -> None:
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(omp_threads)


def _ensure_c_inputs(c_inputs_dir: Path) -> None:
    topol = c_inputs_dir / "topol-cg.mtop"
    ref = c_inputs_dir / "ref-cg.gro"
    if topol.is_file() and ref.is_file():
        return
    setup = BENCH_DIR / "prepare_c_covar_inputs.sh"
    if not setup.is_file():
        raise FileNotFoundError(f"missing setup script: {setup}")
    subprocess.run(["bash", str(setup)], check=True)


def _write_covar_inp(workdir: Path, n_frames: int, n_corr: int) -> Path:
    inp = workdir / "covar.inp"
    inp.write_text(
        f"""#fnTop
topol-cg.mtop
#fnCrd
traj-cg.trr
#fnJob
static.job
#nRead
{n_frames}
#analysisInterval
1
#fnRef
ref-cg.gro
#alignGrp
0
#analyzeGrp
0
#wrap
0
#nCorr
{n_corr}
#winSigma
{WIN_SIGMA}
#binaryMatrix
1
#doGenModes
0
#convergence
1.0e-5
#maxIter
100
#fnOut
cg
"""
    )
    return inp


def _localize_aa_trajectory(paths) -> tuple[Path, Path]:
    local_dir = _tmp_root() / f"hewl_aa_{os.getpid()}"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_tpr = local_dir / paths.aa_topol.name
    local_trj = local_dir / paths.aa_traj.name
    if not local_tpr.exists():
        shutil.copy2(paths.aa_topol, local_tpr)
    if not local_trj.exists():
        shutil.copy2(paths.aa_traj, local_trj)
    return local_tpr, local_trj


def _write_cg_cache(cg: CoarseGrain, u_cg: mda.Universe, cache_dir: Path) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    top_path = cache_dir / "topol-cg.top"
    traj_path = cache_dir / "traj-cg.trr"
    cg.write_topology(u_cg, str(top_path))
    cg.write_trajectory(u_cg, str(traj_path))


def _localize_cg_cache(cache_dir: Path) -> Path:
    local_dir = _tmp_root() / f"hewl_cg_{os.getpid()}"
    if local_dir.exists():
        shutil.rmtree(local_dir)
    shutil.copytree(cache_dir, local_dir)
    return local_dir


def _load_cg_universe(cache_dir: Path) -> tuple[CoarseGrain, mda.Universe]:
    top_path = cache_dir / "topol-cg.top"
    traj_path = cache_dir / "traj-cg.trr"
    map_path = cache_dir / "cg_map.npz"
    if not top_path.is_file() or not traj_path.is_file() or not map_path.is_file():
        raise FileNotFoundError(f"CG cache incomplete under {cache_dir}")
    u_cg = mda.Universe(str(top_path), str(traj_path), topology_format="ITP")
    cg = CoarseGrain.from_cg_map(u_cg.atoms, map_path)
    cg._restore_bead_masses(u_cg)
    return cg, u_cg


def run_cg_reference(n_frames: int, output_dir: Path) -> tuple[dict[str, float], Path]:
    _set_py_thread_env(1)
    paths = resolve_hewl_solution_303K_paths()
    local_tpr, local_trj = _localize_aa_trajectory(paths)

    cg, u_cg = CoarseGrain.cg_universe(
        (local_tpr, local_trj),
        select="all",
        stop=n_frames,
        benchmark=True,
    )
    phases = dict(cg.benchmark or {})

    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / "cg_map.npz"
    _write_cg_cache(cg, u_cg, output_dir)
    cg.save_mapping(map_path)
    return phases, output_dir


def run_fresean_only(
    cg_cache_dir: Path,
    n_jobs: int,
    n_workers: int,
    omp_threads: int,
) -> tuple[float, dict[str, float], CoarseGrain]:
    _set_py_thread_env(omp_threads)
    local_cache = _localize_cg_cache(cg_cache_dir)
    cg, u_cg = _load_cg_universe(local_cache)

    u_cg.trajectory[0]
    ref_cg = u_cg.atoms.positions.copy()
    u_cg.trajectory.add_transformations(
        Align(u_cg.atoms, reference_positions=ref_cg, place_com_in_box=False),
    )

    analysis = FRESEAN(
        u_cg,
        select="all",
        n_constraints=cg.mapping.n_constraints,
        n_corr=N_CORR,
        dt=DT_PS,
        sigma=SIGMA_CM1,
        lag_symmetrization="average",
        n_jobs=n_jobs,
    )

    if n_workers > 1:
        phases = analysis.run(
            n_workers=n_workers,
            backend="multiprocessing",
            verbose=False,
            benchmark=True,
        )
    else:
        phases = analysis.run(verbose=False, benchmark=True)
    bench_t_py_fresean = float(phases[BENCH_T_SPECTRAL])
    _ = analysis.results.freqs.shape
    return bench_t_py_fresean, phases, cg


def run_py_benchmark(
    n_jobs: int,
    n_workers: int,
    n_frames: int,
    omp_threads: int = 1,
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    _set_py_thread_env(omp_threads)
    paths = resolve_hewl_solution_303K_paths()
    local_tpr, local_trj = _localize_aa_trajectory(paths)

    cg, u_cg = CoarseGrain.cg_universe(
        (local_tpr, local_trj),
        select="all",
        stop=n_frames,
        benchmark=True,
    )
    py_coarse_phases = dict(cg.benchmark or {})
    bench_t_py_coarse = float(py_coarse_phases.get(BENCH_T_TOTAL, 0.0))

    u_cg.trajectory[0]
    ref_cg = u_cg.atoms.positions.copy()
    u_cg.trajectory.add_transformations(
        Align(u_cg.atoms, reference_positions=ref_cg, place_com_in_box=False),
    )
    analysis = FRESEAN(
        u_cg,
        select="all",
        n_constraints=cg.mapping.n_constraints,
        n_corr=N_CORR,
        dt=DT_PS,
        sigma=SIGMA_CM1,
        lag_symmetrization="average",
        n_jobs=n_jobs,
    )
    if n_workers > 1:
        py_fresean_phases = analysis.run(
            n_workers=n_workers,
            backend="multiprocessing",
            verbose=False,
            benchmark=True,
        )
    else:
        py_fresean_phases = analysis.run(verbose=False, benchmark=True)
    bench_t_py_fresean = float(py_fresean_phases[BENCH_T_SPECTRAL])
    _ = analysis.results.freqs.shape
    return bench_t_py_coarse, bench_t_py_fresean, py_coarse_phases, py_fresean_phases


def _prepare_c_workdir(c_inputs_dir: Path, workdir: Path) -> None:
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    for name in ("topol-cg.mtop", "ref-cg.gro", "static.job", "traj-cg.trr"):
        src = c_inputs_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing C covar input: {src}")
        dest = workdir / name
        if src.is_symlink():
            dest.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dest)


def run_c_benchmark(
    ncpus: int,
    c_inputs_dir: Path,
    n_frames: int,
    workdir: Path | None = None,
) -> tuple[float, float]:
    local_work = workdir or (_tmp_root() / f"hewl_c_{os.getpid()}")
    _prepare_c_workdir(c_inputs_dir, local_work)
    _write_covar_inp(local_work, n_frames=n_frames, n_corr=N_CORR)

    fresean_bin = str(DEFAULT_FRESEAN_BIN)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(ncpus)

    t0 = time.perf_counter()
    subprocess.run(
        [fresean_bin, "covar", "-f", "covar.inp"],
        cwd=local_work,
        check=True,
        env=env,
    )
    bench_t_c_covar = time.perf_counter() - t0

    mmat = local_work / "covar_cg.mmat"
    if not mmat.is_file():
        raise FileNotFoundError(f"covar did not produce {mmat}")

    t1 = time.perf_counter()
    subprocess.run(
        [fresean_bin, "eigen", "-m", "covar_cg.mmat", "-n", str(N_CORR)],
        cwd=local_work,
        check=True,
        env=env,
    )
    bench_t_c_eigen = time.perf_counter() - t1
    return bench_t_c_covar, bench_t_c_eigen


def _append_summary_csv(csv_path: Path, result: BenchmarkResult) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(result).keys())
    write_header = not csv_path.is_file()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(result))


def _load_cg_reference_total(cg_ref_json: Path) -> float:
    if not cg_ref_json.is_file():
        return 0.0
    data = json.loads(cg_ref_json.read_text())
    coarse = data.get("py_coarse") or {}
    return float(coarse.get(BENCH_T_TOTAL, data.get(BENCH_T_PY_COARSE, 0.0)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["cg", "fresean", "c", "all"],
        default="all",
        help="benchmark phase to run",
    )
    parser.add_argument(
        "--case",
        default="",
        help=f"py case: {', '.join(BENCH_CASES)} (required for fresean mode)",
    )
    parser.add_argument(
        "--ncpus",
        type=int,
        default=int(os.environ.get("NCPUS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
    )
    parser.add_argument(
        "--parallel-config",
        default=None,
        help="legacy alias for --case",
    )
    parser.add_argument("--omp-threads", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--n-frames", type=int, default=N_FRAMES_DEFAULT)
    parser.add_argument("--c-inputs-dir", type=Path, default=DEFAULT_C_INPUTS)
    parser.add_argument("--cg-cache-dir", type=Path, default=DEFAULT_CG_CACHE)
    parser.add_argument("--cg-reference-dir", type=Path, default=DEFAULT_CG_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-c-inputs-check",
        action="store_true",
        help="do not run prepare_c_covar_inputs.sh if inputs missing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ncpus < 1:
        print("ncpus must be >= 1", file=sys.stderr)
        return 2

    case = args.case or ""
    if args.parallel_config:
        case, omp_threads, n_jobs, n_workers = resolve_parallel_config(
            args.parallel_config, args.ncpus
        )
    elif case:
        omp_threads, n_jobs, n_workers = resolve_case(case, args.ncpus)
    else:
        omp_threads = args.omp_threads if args.omp_threads is not None else 1
        n_jobs = args.n_jobs if args.n_jobs is not None else args.ncpus
        n_workers = args.n_workers if args.n_workers is not None else 1
    if args.n_jobs is not None:
        n_jobs = args.n_jobs
    if args.n_workers is not None:
        n_workers = args.n_workers
    if args.omp_threads is not None:
        omp_threads = args.omp_threads
    if n_jobs < 1 or n_workers < 1 or omp_threads < 1:
        print("n_jobs, n_workers, and omp_threads must be >= 1", file=sys.stderr)
        return 2

    if args.mode == "fresean" and not case:
        print("--case is required for --mode fresean", file=sys.stderr)
        return 2

    if args.mode in {"c", "all"} and not args.skip_c_inputs_check:
        _ensure_c_inputs(args.c_inputs_dir)

    if args.output_dir is None:
        if args.mode == "cg":
            args.output_dir = args.cg_reference_dir
        elif args.mode == "c":
            args.output_dir = DEFAULT_C_SPECTRAL
        elif args.mode == "fresean":
            args.output_dir = DEFAULT_PY_CASES / case
        else:
            args.output_dir = DEFAULT_RESULTS / "legacy_all"

    bench_t_py_coarse = 0.0
    bench_t_py_fresean = 0.0
    bench_t_c_covar = 0.0
    bench_t_c_eigen = 0.0
    py_coarse_phases: dict[str, float] | None = None
    py_fresean_phases: dict[str, float] | None = None

    if args.mode == "cg":
        print(f"[py CG reference] n_frames={args.n_frames}")
        py_coarse_phases, cache_dir = run_cg_reference(args.n_frames, args.output_dir)
        bench_t_py_coarse = float(py_coarse_phases[BENCH_T_TOTAL])
        print(f"  bench_t_total: {bench_t_py_coarse:.2f} s")
        print(f"  cache: {cache_dir}")

    elif args.mode == "fresean":
        print(
            f"[py FRESEAN] case={case} ncpus={args.ncpus} "
            f"OMP={omp_threads} n_workers={n_workers} n_jobs={n_jobs}"
        )
        if not args.cg_cache_dir.is_dir():
            print(f"CG cache missing: {args.cg_cache_dir}", file=sys.stderr)
            return 2
        bench_t_py_fresean, py_fresean_phases, _ = run_fresean_only(
            args.cg_cache_dir,
            n_jobs=n_jobs,
            n_workers=n_workers,
            omp_threads=omp_threads,
        )
        bench_t_py_coarse = _load_cg_reference_total(
            args.cg_reference_dir / "result.json"
        )
        print(f"  fresean: {bench_t_py_fresean:.2f} s")

    elif args.mode == "c":
        print(f"[C spectral] ncpus={args.ncpus} OMP_NUM_THREADS={args.ncpus}")
        bench_t_c_covar, bench_t_c_eigen = run_c_benchmark(
            args.ncpus, args.c_inputs_dir, args.n_frames
        )
        print(f"  covar: {bench_t_c_covar:.2f} s")
        print(f"  eigen: {bench_t_c_eigen:.2f} s")

    else:  # all
        print(
            f"[pyfresean all] ncpus={args.ncpus} "
            f"OMP={omp_threads} n_workers={n_workers} n_jobs={n_jobs}"
        )
        bench_t_py_coarse, bench_t_py_fresean, py_coarse_phases, py_fresean_phases = (
            run_py_benchmark(
                n_jobs=n_jobs,
                n_workers=n_workers,
                n_frames=args.n_frames,
                omp_threads=omp_threads,
            )
        )
        print(f"  coarse: {bench_t_py_coarse:.2f} s")
        print(f"  fresean: {bench_t_py_fresean:.2f} s")
        print(f"[C] ncpus={args.ncpus}")
        bench_t_c_covar, bench_t_c_eigen = run_c_benchmark(
            args.ncpus, args.c_inputs_dir, args.n_frames
        )
        print(f"  covar: {bench_t_c_covar:.2f} s")
        print(f"  eigen: {bench_t_c_eigen:.2f} s")

    result = BenchmarkResult(
        mode=args.mode,
        case=case,
        ncpus=args.ncpus,
        omp_threads=omp_threads,
        py_n_jobs=n_jobs,
        py_n_workers=n_workers,
        hostname=subprocess.check_output(["hostname"], text=True).strip(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        bench_t_py_coarse=bench_t_py_coarse,
        bench_t_py_fresean=bench_t_py_fresean,
        bench_t_py_total=bench_t_py_coarse + bench_t_py_fresean,
        bench_t_c_covar=bench_t_c_covar,
        bench_t_c_eigen=bench_t_c_eigen,
        bench_t_c_total=bench_t_c_covar + bench_t_c_eigen,
        py_coarse=py_coarse_phases,
        py_fresean=py_fresean_phases,
        n_frames=args.n_frames,
        n_corr=N_CORR,
        dt_ps=DT_PS,
    )

    if args.mode == "cg":
        run_dir = args.output_dir
    else:
        run_dir = args.output_dir / f"ncpus_{args.ncpus}"
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "result.json"
    json_path.write_text(json.dumps(asdict(result), indent=2) + "\n")
    _append_summary_csv(args.output_dir / "summary.csv", result)

    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

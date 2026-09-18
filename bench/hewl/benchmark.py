#!/usr/bin/env python3
"""Benchmark pyfresean vs FRESEAN COARSE (C) on HEWL.

Modes (run separately for clean sweeps):
  py_cg       — pyfresean coarse-grain once → cg_reference/pyfresean/
  c_cg        — C fresean coarse once       → cg_reference/c_ref/
  c_aa        — C all-atom covar inputs once → aa_reference/c_ref/
  py_fresean  — py spectral only (CG cache or all-atom trajectory)
  c_spectral  — C covar+eigen only (reads c_ref inputs)
  all         — py CG + py FRESEAN + C spectral in one process (legacy)

Use ``--system cg`` (default) or ``--system aa``. Results land under
``results/results_{cg,aa}/{py_cases,c_spectral,plots}/``.

Legacy aliases: cg → py_cg, fresean → py_fresean, c → c_spectral.
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
    BENCH_T_C_COARSE,
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
from pyfresean.parallel import set_blas_threads
from pyfresean.tests.c_ref.ala_dipeptide_gas_300K import N_CORR, SIGMA_CM1
from pyfresean.tests.c_ref.hewl_solution_303K import (
    DT_PS,
    N_FRAMES_DEFAULT,
    resolve_hewl_solution_303K_paths,
)

BENCH_DIR = Path(__file__).resolve().parent
PYFRESEAN_ROOT = BENCH_DIR.parent.parent
DEFAULT_CG_REFERENCE_ROOT = BENCH_DIR / "cg_reference"
DEFAULT_AA_REFERENCE_ROOT = BENCH_DIR / "aa_reference"
DEFAULT_C_INPUTS = DEFAULT_CG_REFERENCE_ROOT / "c_ref"
DEFAULT_AA_INPUTS = DEFAULT_AA_REFERENCE_ROOT / "c_ref"
DEFAULT_PY_CG_INPUTS = DEFAULT_CG_REFERENCE_ROOT / "pyfresean"
DEFAULT_RESULTS_ROOT = BENCH_DIR / "results"
DEFAULT_RESULTS_CG = DEFAULT_RESULTS_ROOT / "results_cg"
DEFAULT_RESULTS_AA = DEFAULT_RESULTS_ROOT / "results_aa"
DEFAULT_CG_REFERENCE = DEFAULT_PY_CG_INPUTS
DEFAULT_CG_CACHE = DEFAULT_PY_CG_INPUTS
DEFAULT_FRESEAN_BIN = os.environ.get("FRESEAN_BIN", "fresean")
DEFAULT_GMX = os.environ.get("GMX", "gmx")
WIN_SIGMA = 10.0
N_CONSTRAINTS_AA = 6
CPU_COUNTS = (1, 2, 4, 8, 16, 32, 48)
SYSTEMS = ("cg", "aa")

MODES = ("py_cg", "c_cg", "c_aa", "py_fresean", "c_spectral", "all")
MODE_ALIASES = {
    "cg": "py_cg",
    "fresean": "py_fresean",
    "c": "c_spectral",
}

CaseResolver = Callable[[int], tuple[int, int, int]]

BENCH_CASES: dict[str, CaseResolver] = {
    "omp1_njobs_n_nworkers_1": lambda n: (1, n, 1),
    "omp1_njobs_n_nworkers_2": lambda n: (1, n, min(2, n)),
    "omp1_njobs_n_nworkers_n": lambda n: (1, n, n),
    "omp_n_njobs_1": lambda n: (n, 1, 1),
    "omp_n_njobs_1_vec": lambda n: (n, 1, 1),
    "omp1_njobs_n_vec": lambda n: (1, n, 1),
    "omp_hybrid_vec": lambda n: (n, n, 1),
}

_HYBRID_PHASE_CASES = frozenset({"omp1_njobs_n_vec", "omp_hybrid_vec"})


def fresean_parallel_for_case(
    omp_threads: int,
    n_jobs: int,
    case: str,
) -> dict[str, dict[str, int]]:
    """Map a HEWL benchmark case to a FRESEAN ``parallel`` dict.

    ``omp1_njobs_n_vec`` and ``omp_hybrid_vec`` use a tile thread pool for
    ``corr_matrix`` only (``omp_threads=1`` there) while keeping BLAS threads
    for ``velocity_fft`` and ``eigen``. ``omp_hybrid_vec`` sets
    ``omp_threads=N`` on those phases; ``omp1_njobs_n_vec`` leaves them at 1.
    Other cases apply the same ``omp_threads`` / ``n_jobs`` to ``corr_matrix``
    and ``eigen``.
    """
    omp_threads = max(1, int(omp_threads))
    n_jobs = max(1, int(n_jobs))

    if case in _HYBRID_PHASE_CASES:
        return {
            "velocity_fft": {"n_jobs": 1, "omp_threads": omp_threads},
            "corr_matrix": {"n_jobs": n_jobs, "omp_threads": 1},
            "eigen": {"n_jobs": 1, "omp_threads": omp_threads},
        }

    return {
        "velocity_fft": {"n_jobs": 1, "omp_threads": omp_threads},
        "corr_matrix": {"n_jobs": n_jobs, "omp_threads": omp_threads},
        "eigen": {"n_jobs": n_jobs, "omp_threads": omp_threads},
    }


CG_MODES = frozenset({"py_cg", "c_cg"})
AA_PREP_MODES = frozenset({"c_aa"})
INPUT_PREP_MODES = CG_MODES | AA_PREP_MODES


def results_root(system: str) -> Path:
    if system == "aa":
        return DEFAULT_RESULTS_AA
    if system == "cg":
        return DEFAULT_RESULTS_CG
    raise ValueError(f"unknown system {system!r}; choose: {', '.join(SYSTEMS)}")


def default_c_spectral_dir(system: str) -> Path:
    return results_root(system) / "c_spectral"


def default_py_cases_dir(system: str) -> Path:
    return results_root(system) / "py_cases"


def default_c_inputs_dir(system: str) -> Path:
    if system == "aa":
        return DEFAULT_AA_INPUTS
    return DEFAULT_C_INPUTS


@dataclass
class BenchmarkResult:
    mode: str
    system: str
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
    bench_t_c_coarse: float
    bench_t_c_covar: float
    bench_t_c_eigen: float
    bench_t_c_total: float
    py_coarse: dict[str, float] | None
    py_fresean: dict[str, float] | None
    c_coarse: dict[str, float] | None
    n_frames: int
    n_corr: int
    dt_ps: float


def normalize_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


def _tmp_root() -> Path:
    root = Path(
        os.environ.get("SLURM_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_case(case: str, ncpus: int) -> tuple[int, int, int]:
    if case not in BENCH_CASES:
        raise ValueError(
            f"unknown case {case!r}; choose: {', '.join(BENCH_CASES)}"
        )
    return BENCH_CASES[case](ncpus)


def resolve_parallel_config(
    config: str, ncpus: int
) -> tuple[str, int, int, int]:
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


def _c_cg_inputs_ready(output_dir: Path) -> bool:
    return (output_dir / "topol-cg.mtop").is_file() and (
        output_dir / "ref-cg.gro"
    ).is_file()


def _aa_c_inputs_ready(output_dir: Path) -> bool:
    return (
        (output_dir / "topol-aa.mtop").is_file()
        and (output_dir / "ref-aa.gro").is_file()
        and (output_dir / "aa.trr").is_file()
    )


def run_c_cg_reference(
    n_frames: int,
    output_dir: Path,
    *,
    force: bool = False,
) -> tuple[float, dict[str, float]]:
    if not force and _c_cg_inputs_ready(output_dir):
        print(f"C CG inputs already present: {output_dir}")
        result_json = output_dir / "result.json"
        if result_json.is_file():
            data = json.loads(result_json.read_text())
            phases = data.get("c_coarse") or {}
            total = float(
                data.get(BENCH_T_C_COARSE, phases.get(BENCH_T_TOTAL, 0.0))
            )
            return total, phases
        return 0.0, {}

    paths = resolve_hewl_solution_303K_paths()
    for path in (paths.aa_topol, paths.aa_traj, paths.topol_prot):
        if not path.is_file():
            raise FileNotFoundError(f"missing HEWL input: {path}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    work = output_dir
    fresean_bin = DEFAULT_FRESEAN_BIN
    gmx = DEFAULT_GMX
    phases: dict[str, float] = {}

    shutil.copy2(paths.script_dir / "static.job", work / "static.job")

    t_mtop = time.perf_counter()
    subprocess.run(
        [
            fresean_bin,
            "mtop",
            "-p",
            str(paths.topol_prot),
        ],
        cwd=work,
        input=f"{paths.topol_prot}\n{paths.topol_prot}\ntopol-aa.mtop\n",
        text=True,
        check=True,
    )
    phases["bench_t_mtop"] = time.perf_counter() - t_mtop

    trj_end = n_frames * DT_PS
    t_trjconv = time.perf_counter()
    subprocess.run(
        [
            gmx,
            "trjconv",
            "-s",
            str(paths.aa_topol),
            "-f",
            str(paths.aa_traj),
            "-o",
            "aa.trr",
            "-b",
            "0",
            "-e",
            str(trj_end),
        ],
        cwd=work,
        input="0\n",
        text=True,
        check=True,
    )
    phases["bench_t_trjconv"] = time.perf_counter() - t_trjconv

    coarse_inp = work / "coarse.inp"
    coarse_inp.write_text(f"""#fnTop
topol-aa.mtop
#fnCrd
aa.trr
#fnVel
#fnJob
static.job
#grp
0
#nRead
{n_frames}
#nSample
1
#fnOutTraj
tmptraj.gro
#fnOutTopol
topol-cg.mtop
""")

    t_coarse = time.perf_counter()
    subprocess.run(
        [fresean_bin, "coarse", "-f", "coarse.inp"],
        cwd=work,
        check=True,
    )
    phases["bench_t_coarse"] = time.perf_counter() - t_coarse

    tmptraj = work / "tmptraj.gro"
    n_atoms = int(tmptraj.read_text().splitlines()[1].strip())
    n_lines = n_atoms + 3
    (work / "ref-cg.gro").write_text(
        "".join(tmptraj.read_text().splitlines(True)[:n_lines])
    )

    for name in ("tmptraj.gro", "aa.trr", "topol-aa.mtop", "coarse.inp"):
        (work / name).unlink(missing_ok=True)

    traj_link = work / "traj-cg.trr"
    if traj_link.exists() or traj_link.is_symlink():
        traj_link.unlink()
    traj_src = paths.data_dir / "traj-cg.trr"
    traj_link.symlink_to(traj_src.resolve())

    phases[BENCH_T_TOTAL] = (
        phases["bench_t_mtop"]
        + phases["bench_t_trjconv"]
        + phases["bench_t_coarse"]
    )
    return phases[BENCH_T_TOTAL], phases


def run_c_aa_reference(
    n_frames: int,
    output_dir: Path,
    *,
    force: bool = False,
) -> tuple[float, dict[str, float]]:
    if not force and _aa_c_inputs_ready(output_dir):
        print(f"C AA inputs already present: {output_dir}")
        result_json = output_dir / "result.json"
        if result_json.is_file():
            data = json.loads(result_json.read_text())
            phases = data.get("c_coarse") or {}
            total = float(
                data.get(BENCH_T_C_COARSE, phases.get(BENCH_T_TOTAL, 0.0))
            )
            return total, phases
        return 0.0, {}

    paths = resolve_hewl_solution_303K_paths()
    for path in (paths.aa_topol, paths.aa_traj, paths.topol_prot):
        if not path.is_file():
            raise FileNotFoundError(f"missing HEWL input: {path}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    work = output_dir
    fresean_bin = DEFAULT_FRESEAN_BIN
    gmx = DEFAULT_GMX
    phases: dict[str, float] = {}

    shutil.copy2(paths.script_dir / "static.job", work / "static.job")

    t_mtop = time.perf_counter()
    subprocess.run(
        [
            fresean_bin,
            "mtop",
            "-p",
            str(paths.topol_prot),
        ],
        cwd=work,
        input=f"{paths.topol_prot}\n{paths.topol_prot}\ntopol-aa.mtop\n",
        text=True,
        check=True,
    )
    phases["bench_t_mtop"] = time.perf_counter() - t_mtop

    trj_end = n_frames * DT_PS
    t_trjconv = time.perf_counter()
    subprocess.run(
        [
            gmx,
            "trjconv",
            "-s",
            str(paths.aa_topol),
            "-f",
            str(paths.aa_traj),
            "-o",
            "aa.trr",
            "-b",
            "0",
            "-e",
            str(trj_end),
        ],
        cwd=work,
        input="0\n",
        text=True,
        check=True,
    )
    phases["bench_t_trjconv"] = time.perf_counter() - t_trjconv

    t_ref = time.perf_counter()
    subprocess.run(
        [
            gmx,
            "trjconv",
            "-s",
            str(paths.aa_topol),
            "-f",
            "aa.trr",
            "-dump",
            "0",
            "-o",
            "ref-aa.gro",
        ],
        cwd=work,
        input="0\n",
        text=True,
        check=True,
    )
    phases["bench_t_ref"] = time.perf_counter() - t_ref
    phases[BENCH_T_TOTAL] = (
        phases["bench_t_mtop"]
        + phases["bench_t_trjconv"]
        + phases["bench_t_ref"]
    )
    return phases[BENCH_T_TOTAL], phases


def _ensure_c_inputs(
    c_inputs_dir: Path,
    n_frames: int,
    *,
    system: str = "cg",
    force: bool = False,
) -> None:
    if system == "aa":
        if not force and _aa_c_inputs_ready(c_inputs_dir):
            return
        run_c_aa_reference(n_frames, c_inputs_dir, force=force)
        return
    if not force and _c_cg_inputs_ready(c_inputs_dir):
        return
    run_c_cg_reference(n_frames, c_inputs_dir, force=force)


def _write_covar_inp(
    workdir: Path,
    n_frames: int,
    n_corr: int,
    *,
    system: str = "cg",
) -> Path:
    if system == "aa":
        topol = "topol-aa.mtop"
        traj = "aa.trr"
        ref = "ref-aa.gro"
        out_stem = "aa"
    else:
        topol = "topol-cg.mtop"
        traj = "traj-cg.trr"
        ref = "ref-cg.gro"
        out_stem = "cg"
    inp = workdir / "covar.inp"
    inp.write_text(f"""#fnTop
{topol}
#fnCrd
{traj}
#fnJob
static.job
#nRead
{n_frames}
#analysisInterval
1
#fnRef
{ref}
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
{out_stem}
""")
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


def _write_cg_cache(
    cg: CoarseGrain, u_cg: mda.Universe, cache_dir: Path
) -> None:
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
    if (
        not top_path.is_file()
        or not traj_path.is_file()
        or not map_path.is_file()
    ):
        raise FileNotFoundError(f"CG cache incomplete under {cache_dir}")
    u_cg = mda.Universe(str(top_path), str(traj_path), topology_format="ITP")
    cg = CoarseGrain.from_cg_map(u_cg.atoms, map_path)
    cg._restore_bead_masses(u_cg)
    return cg, u_cg


def run_py_cg_reference(
    n_frames: int, output_dir: Path
) -> tuple[dict[str, float], Path]:
    set_blas_threads(1)
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


def run_py_fresean_aa_only(
    n_frames: int,
    n_jobs: int,
    n_workers: int,
    omp_threads: int,
    case: str = "omp_n_njobs_1",
) -> tuple[float, dict[str, float]]:
    paths = resolve_hewl_solution_303K_paths()
    local_tpr, local_trj = _localize_aa_trajectory(paths)
    u_aa = mda.Universe(str(local_tpr), str(local_trj))

    u_aa.trajectory[0]
    ref_aa = u_aa.atoms.positions.copy()
    u_aa.trajectory.add_transformations(
        Align(u_aa.atoms, reference_positions=ref_aa, place_com_in_box=False),
    )

    analysis = FRESEAN(
        u_aa,
        select="all",
        n_constraints=N_CONSTRAINTS_AA,
        n_corr=N_CORR,
        dt=DT_PS,
        sigma=SIGMA_CM1,
        lag_symmetrization="average",
        parallel=fresean_parallel_for_case(omp_threads, n_jobs, case),
    )

    run_kwargs = {"verbose": False, "benchmark": True, "stop": n_frames}
    if n_workers > 1:
        phases = analysis.run(
            n_workers=n_workers,
            backend="multiprocessing",
            **run_kwargs,
        )
    else:
        phases = analysis.run(**run_kwargs)
    bench_t_py_fresean = float(phases[BENCH_T_SPECTRAL])
    _ = analysis.results.freqs.shape
    return bench_t_py_fresean, phases


def run_py_fresean_only(
    cg_cache_dir: Path,
    n_jobs: int,
    n_workers: int,
    omp_threads: int,
    case: str = "omp_n_njobs_1",
) -> tuple[float, dict[str, float], CoarseGrain]:
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
        parallel=fresean_parallel_for_case(omp_threads, n_jobs, case),
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
    case: str = "omp_n_njobs_1",
) -> tuple[float, float, dict[str, float], dict[str, float]]:
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
        parallel=fresean_parallel_for_case(omp_threads, n_jobs, case),
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
    return (
        bench_t_py_coarse,
        bench_t_py_fresean,
        py_coarse_phases,
        py_fresean_phases,
    )


def _prepare_c_workdir(
    c_inputs_dir: Path, workdir: Path, *, system: str = "cg"
) -> None:
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    if system == "aa":
        names = ("topol-aa.mtop", "ref-aa.gro", "static.job", "aa.trr")
    else:
        names = ("topol-cg.mtop", "ref-cg.gro", "static.job", "traj-cg.trr")
    for name in names:
        src = c_inputs_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing C covar input: {src}")
        dest = workdir / name
        if src.is_symlink():
            dest.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dest)


def run_c_spectral(
    ncpus: int,
    c_inputs_dir: Path,
    n_frames: int,
    workdir: Path | None = None,
    *,
    system: str = "cg",
) -> tuple[float, float]:
    out_stem = "aa" if system == "aa" else "cg"
    local_work = workdir or (_tmp_root() / f"hewl_c_{system}_{os.getpid()}")
    _prepare_c_workdir(c_inputs_dir, local_work, system=system)
    _write_covar_inp(
        local_work, n_frames=n_frames, n_corr=N_CORR, system=system
    )

    fresean_bin = DEFAULT_FRESEAN_BIN
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

    mmat = local_work / f"covar_{out_stem}.mmat"
    if not mmat.is_file():
        raise FileNotFoundError(f"covar did not produce {mmat}")

    t1 = time.perf_counter()
    subprocess.run(
        [
            fresean_bin,
            "eigen",
            "-m",
            f"covar_{out_stem}.mmat",
            "-n",
            str(N_CORR),
        ],
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


def _load_py_cg_reference_total(cg_ref_json: Path) -> float:
    if not cg_ref_json.is_file():
        return 0.0
    data = json.loads(cg_ref_json.read_text())
    coarse = data.get("py_coarse") or {}
    return float(coarse.get(BENCH_T_TOTAL, data.get(BENCH_T_PY_COARSE, 0.0)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[*MODES, *MODE_ALIASES.keys()],
        default="all",
        help="benchmark phase to run",
    )
    parser.add_argument(
        "--case",
        default="",
        help=f"py case: {', '.join(BENCH_CASES)} (required for py_fresean mode)",
    )
    parser.add_argument(
        "--ncpus",
        type=int,
        default=int(
            os.environ.get("NCPUS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        ),
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
    parser.add_argument(
        "--system",
        choices=SYSTEMS,
        default="cg",
        help="cg: coarse-grained spectral; aa: all-atom spectral",
    )
    parser.add_argument("--c-inputs-dir", type=Path, default=None)
    parser.add_argument("--cg-cache-dir", type=Path, default=DEFAULT_CG_CACHE)
    parser.add_argument(
        "--cg-reference-dir", type=Path, default=DEFAULT_CG_REFERENCE
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-c-inputs-check",
        action="store_true",
        help="do not build c_ref inputs if missing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate CG reference outputs even if cache/inputs exist",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode = normalize_mode(args.mode)
    if mode not in MODES:
        print(f"unknown mode {args.mode!r}", file=sys.stderr)
        return 2
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
        print(
            "n_jobs, n_workers, and omp_threads must be >= 1", file=sys.stderr
        )
        return 2

    if mode == "py_fresean" and not case:
        print("--case is required for --mode py_fresean", file=sys.stderr)
        return 2

    if mode in {"py_cg", "c_cg"} and args.system != "cg":
        print(
            f"--system {args.system!r} is invalid for --mode {mode}",
            file=sys.stderr,
        )
        return 2
    if mode == "c_aa" and args.system != "aa":
        args.system = "aa"

    if args.c_inputs_dir is None:
        args.c_inputs_dir = default_c_inputs_dir(args.system)

    if mode in {"c_spectral", "all"} and not args.skip_c_inputs_check:
        _ensure_c_inputs(
            args.c_inputs_dir,
            args.n_frames,
            system=args.system,
            force=args.force,
        )

    if args.output_dir is None:
        if mode == "py_cg":
            args.output_dir = args.cg_reference_dir
        elif mode == "c_cg":
            args.output_dir = DEFAULT_C_INPUTS
        elif mode == "c_aa":
            args.output_dir = DEFAULT_AA_INPUTS
        elif mode == "c_spectral":
            args.output_dir = default_c_spectral_dir(args.system)
        elif mode == "py_fresean":
            args.output_dir = default_py_cases_dir(args.system) / case
        else:
            args.output_dir = results_root(args.system) / "legacy_all"

    bench_t_py_coarse = 0.0
    bench_t_py_fresean = 0.0
    bench_t_c_coarse = 0.0
    bench_t_c_covar = 0.0
    bench_t_c_eigen = 0.0
    py_coarse_phases: dict[str, float] | None = None
    py_fresean_phases: dict[str, float] | None = None
    c_coarse_phases: dict[str, float] | None = None

    if mode == "py_cg":
        print(f"[py CG] n_frames={args.n_frames}")
        py_coarse_phases, cache_dir = run_py_cg_reference(
            args.n_frames, args.output_dir
        )
        bench_t_py_coarse = float(py_coarse_phases[BENCH_T_TOTAL])
        print(f"  bench_t_total: {bench_t_py_coarse:.2f} s")
        print(f"  cache: {cache_dir}")

    elif mode == "c_cg":
        print(f"[C CG] n_frames={args.n_frames}")
        bench_t_c_coarse, c_coarse_phases = run_c_cg_reference(
            args.n_frames,
            args.output_dir,
            force=args.force,
        )
        print(f"  bench_t_coarse: {bench_t_c_coarse:.2f} s")
        print(f"  outputs: {args.output_dir}")

    elif mode == "c_aa":
        print(f"[C AA inputs] n_frames={args.n_frames}")
        bench_t_c_coarse, c_coarse_phases = run_c_aa_reference(
            args.n_frames,
            args.output_dir,
            force=args.force,
        )
        print(f"  bench_t_prep: {bench_t_c_coarse:.2f} s")
        print(f"  outputs: {args.output_dir}")

    elif mode == "py_fresean":
        print(
            f"[py FRESEAN] system={args.system} case={case} ncpus={args.ncpus} "
            f"OMP={omp_threads} n_workers={n_workers} n_jobs={n_jobs}"
        )
        if args.system == "aa":
            bench_t_py_fresean, py_fresean_phases = run_py_fresean_aa_only(
                args.n_frames,
                n_jobs=n_jobs,
                n_workers=n_workers,
                omp_threads=omp_threads,
                case=case,
            )
        else:
            if not args.cg_cache_dir.is_dir():
                print(f"CG cache missing: {args.cg_cache_dir}", file=sys.stderr)
                return 2
            bench_t_py_fresean, py_fresean_phases, _ = run_py_fresean_only(
                args.cg_cache_dir,
                n_jobs=n_jobs,
                n_workers=n_workers,
                omp_threads=omp_threads,
                case=case,
            )
            bench_t_py_coarse = _load_py_cg_reference_total(
                args.cg_reference_dir / "result.json"
            )
        print(f"  fresean: {bench_t_py_fresean:.2f} s")

    elif mode == "c_spectral":
        print(
            f"[C spectral] system={args.system} ncpus={args.ncpus} "
            f"OMP_NUM_THREADS={args.ncpus}"
        )
        bench_t_c_covar, bench_t_c_eigen = run_c_spectral(
            args.ncpus,
            args.c_inputs_dir,
            args.n_frames,
            system=args.system,
        )
        print(f"  covar: {bench_t_c_covar:.2f} s")
        print(f"  eigen: {bench_t_c_eigen:.2f} s")

    else:  # all
        print(
            f"[pyfresean all] ncpus={args.ncpus} "
            f"OMP={omp_threads} n_workers={n_workers} n_jobs={n_jobs}"
        )
        (
            bench_t_py_coarse,
            bench_t_py_fresean,
            py_coarse_phases,
            py_fresean_phases,
        ) = run_py_benchmark(
            n_jobs=n_jobs,
            n_workers=n_workers,
            n_frames=args.n_frames,
            omp_threads=omp_threads,
            case=case,
        )
        print(f"  coarse: {bench_t_py_coarse:.2f} s")
        print(f"  fresean: {bench_t_py_fresean:.2f} s")
        print(f"[C spectral] ncpus={args.ncpus}")
        bench_t_c_covar, bench_t_c_eigen = run_c_spectral(
            args.ncpus,
            args.c_inputs_dir,
            args.n_frames,
            system=args.system,
        )
        print(f"  covar: {bench_t_c_covar:.2f} s")
        print(f"  eigen: {bench_t_c_eigen:.2f} s")

    result = BenchmarkResult(
        mode=mode,
        system=args.system,
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
        bench_t_c_coarse=bench_t_c_coarse,
        bench_t_c_covar=bench_t_c_covar,
        bench_t_c_eigen=bench_t_c_eigen,
        bench_t_c_total=bench_t_c_covar + bench_t_c_eigen,
        py_coarse=py_coarse_phases,
        py_fresean=py_fresean_phases,
        c_coarse=c_coarse_phases,
        n_frames=args.n_frames,
        n_corr=N_CORR,
        dt_ps=DT_PS,
    )

    if mode in INPUT_PREP_MODES:
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

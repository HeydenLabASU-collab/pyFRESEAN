#!/usr/bin/env python3
"""Merge HEWL benchmark JSON results and plot wall times."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from pyfresean.benchmark_keys import (
    BENCH_T_ASSEMBLE_UNIVERSE,
    BENCH_T_C_COVAR,
    BENCH_T_C_EIGEN,
    BENCH_T_C_TOTAL,
    BENCH_T_CORR_MATRIX,
    BENCH_T_EIGEN,
    BENCH_T_FRAME_PROCESSING,
    BENCH_T_MAPPING,
    BENCH_T_PY_COARSE,
    BENCH_T_PY_FRESEAN,
    BENCH_T_PY_TOTAL,
    BENCH_T_SPECTRAL,
    BENCH_T_TOTAL,
    BENCH_T_VELOCITY_MATRIX,
    BENCH_T_WRITE_OUTPUTS,
)

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = BENCH_DIR / "results"
CG_REF = BENCH_DIR / "cg_reference" / "pyfresean"
CPU_COUNTS = (1, 2, 4, 8, 16, 32, 48)
SYSTEMS = ("cg", "aa")

CASE_LABELS = {
    "omp1_njobs_n_nworkers_1": "OMP=1, n_jobs=N, n_workers=1",
    "omp1_njobs_n_nworkers_2": "OMP=1, n_jobs=N, n_workers=min(2,N)",
    "omp1_njobs_n_nworkers_n": "OMP=1, n_jobs=N, n_workers=N",
    "omp_n_njobs_1": "OMP=N, n_jobs=1, n_workers=1",
    "omp_n_njobs_1_vec": "OMP=N, n_jobs=1, n_workers=1 (vectorized)",
}

LEGACY_BENCH_KEYS = {
    BENCH_T_PY_FRESEAN: "py_fresean_s",
    BENCH_T_PY_COARSE: "py_coarse_s",
    BENCH_T_PY_TOTAL: "py_total_s",
    BENCH_T_C_COVAR: "c_covar_s",
    BENCH_T_C_EIGEN: "c_eigen_s",
    BENCH_T_C_TOTAL: "c_total_s",
}


def results_root(system: str) -> Path:
    return RESULTS_ROOT / f"results_{system}"


def c_spectral_dir(system: str) -> Path:
    return results_root(system) / "c_spectral"


def py_cases_dir(system: str) -> Path:
    return results_root(system) / "py_cases"


def plots_dir(system: str) -> Path:
    return results_root(system) / "plots"


def bench_value(row: dict | None, key: str) -> float:
    if not row:
        return 0.0
    if key in row and row[key] is not None:
        return float(row[key])
    legacy = LEGACY_BENCH_KEYS.get(key)
    if legacy and legacy in row and row[legacy] is not None:
        return float(row[legacy])
    return 0.0


def nested_bench(row: dict, section: str, key: str) -> float:
    block = row.get(section) or {}
    if isinstance(block, dict) and key in block:
        return float(block[key])
    return 0.0


def load_results(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    direct = results_dir / "result.json"
    if direct.is_file():
        rows.append(json.loads(direct.read_text()))
    for path in sorted(results_dir.glob("ncpus_*/result.json")):
        rows.append(json.loads(path.read_text()))
    rows.sort(key=lambda r: int(r["ncpus"]))
    return rows


def py_fresean_time(row: dict) -> float:
    return bench_value(row, BENCH_T_PY_FRESEAN) or nested_bench(
        row, "py_fresean", BENCH_T_SPECTRAL
    )


def c_spectral_time(row: dict | None) -> float:
    if not row:
        return 0.0
    return bench_value(row, BENCH_T_C_TOTAL) or (
        bench_value(row, BENCH_T_C_COVAR) + bench_value(row, BENCH_T_C_EIGEN)
    )


def write_summary(rows: list[dict], out_csv: Path) -> None:
    if not rows:
        return
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _setup_log_axis(ax, ncpus: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ncpus)
    ax.set_xticklabels([str(n) for n in ncpus])
    ax.grid(True, alpha=0.3)


def plot_total_overview(
    case_rows: dict[str, list[dict]],
    c_rows: list[dict],
    plot_path: Path,
    *,
    system: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ncpus_ref: list[int] | None = None
    c_map = {int(r["ncpus"]): r for r in c_rows}

    for case, rows in case_rows.items():
        if not rows:
            continue
        ncpus = [int(r["ncpus"]) for r in rows]
        ncpus_ref = ncpus
        totals = [py_fresean_time(r) for r in rows]
        ax.plot(ncpus, totals, "o-", label=CASE_LABELS.get(case, case))

    if ncpus_ref and c_map:
        c_ncpus = [n for n in ncpus_ref if n in c_map]
        if c_ncpus:
            ax.plot(
                c_ncpus,
                [c_spectral_time(c_map[n]) for n in c_ncpus],
                "s--",
                color="black",
                linewidth=2,
                label="C covar+eigen",
            )

    ax.set_xlabel("CPUs")
    ax.set_ylabel("wall time (s)")
    ax.set_title(
        f"HEWL {system.upper()} spectral time — pyfresean FRESEAN vs C covar+eigen"
    )
    _setup_log_axis(ax, ncpus_ref or list(CPU_COUNTS))
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def plot_case_total(
    case: str,
    rows: list[dict],
    c_rows: list[dict],
    plot_path: Path,
    *,
    system: str,
) -> None:
    import matplotlib.pyplot as plt

    ncpus = [int(r["ncpus"]) for r in rows]
    c_map = {int(r["ncpus"]): r for r in c_rows}
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(
        ncpus,
        [py_fresean_time(r) for r in rows],
        "o-",
        label="pyfresean FRESEAN",
    )
    if c_map:
        ax.plot(
            ncpus,
            [c_spectral_time(c_map.get(n)) for n in ncpus],
            "s-",
            label="C covar+eigen",
        )

    ax.set_xlabel("CPUs")
    ax.set_ylabel("wall time (s)")
    ax.set_title(
        f"HEWL {system.upper()} spectral — {CASE_LABELS.get(case, case)}"
    )
    _setup_log_axis(ax, ncpus)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def plot_case_breakdown(
    case: str,
    rows: list[dict],
    plot_path: Path,
    *,
    system: str,
) -> None:
    import matplotlib.pyplot as plt

    ncpus = [int(r["ncpus"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        ncpus,
        [nested_bench(r, "py_fresean", BENCH_T_VELOCITY_MATRIX) for r in rows],
        "o-",
        label="velocity matrix",
    )
    ax.plot(
        ncpus,
        [nested_bench(r, "py_fresean", BENCH_T_CORR_MATRIX) for r in rows],
        "s-",
        label="corr matrix",
    )
    ax.plot(
        ncpus,
        [nested_bench(r, "py_fresean", BENCH_T_EIGEN) for r in rows],
        "d-",
        label="eigen",
    )
    ax.plot(
        ncpus,
        [py_fresean_time(r) for r in rows],
        "k--",
        linewidth=1.5,
        label="FRESEAN total",
    )
    ax.set_title(
        f"pyfresean {system.upper()} breakdown — {CASE_LABELS.get(case, case)}"
    )
    ax.set_xlabel("CPUs")
    ax.set_ylabel("wall time (s)")
    _setup_log_axis(ax, ncpus)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def plot_cg_reference(plot_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = load_results(CG_REF)
    if not rows:
        return
    row = rows[0]
    phases = row.get("py_coarse") or {}
    keys = [
        BENCH_T_MAPPING,
        BENCH_T_FRAME_PROCESSING,
        BENCH_T_WRITE_OUTPUTS,
        BENCH_T_ASSEMBLE_UNIVERSE,
    ]
    values = [float(phases.get(k, 0.0)) for k in keys]
    labels = ["mapping", "frame proc", "write", "assemble"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    ax.set_ylabel("wall time (s)")
    ax.set_title(f"py CG reference (bench_t_total={sum(values):.1f}s)")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def print_table(
    system: str, case: str, rows: list[dict], c_rows: list[dict]
) -> None:
    print(f"\n=== {system.upper()} — {CASE_LABELS.get(case, case)} ===")
    print(f"{'ncpus':>5}  {'bench_t_py_fresean':>18}  {'bench_t_c_total':>16}")
    c_map = {int(r["ncpus"]): r for r in c_rows}
    for r in rows:
        n = int(r["ncpus"])
        py_f = py_fresean_time(r)
        c_total = c_spectral_time(c_map.get(n))
        print(f"{n:5d}  {py_f:18.2f}  {c_total:16.2f}")


def collect_system(
    system: str,
    cases: list[str],
    *,
    plot: bool,
) -> dict[str, list[dict]]:
    c_rows = load_results(c_spectral_dir(system))
    case_rows: dict[str, list[dict]] = {}
    plots = plots_dir(system)

    for case in cases:
        case_dir = py_cases_dir(system) / case
        rows = load_results(case_dir)
        case_rows[case] = rows
        if not rows:
            print(f"No results in {case_dir}", file=sys.stderr)
            continue
        write_summary(rows, case_dir / "summary.csv")
        print_table(system, case, rows, c_rows)
        if plot:
            plot_case_total(
                case,
                rows,
                c_rows,
                plots / case / "total_vs_c.png",
                system=system,
            )
            plot_case_breakdown(
                case,
                rows,
                plots / case / "breakdown.png",
                system=system,
            )
            print(f"Wrote {plots / case}")

    if plot and case_rows:
        plot_total_overview(
            case_rows,
            c_rows,
            plots / "total_all_cases.png",
            system=system,
        )
        if system == "cg":
            plot_cg_reference(plots / "cg_reference_breakdown.png")
        print(f"Wrote {plots / 'total_all_cases.png'}")

    return case_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=[*SYSTEMS, "all"],
        default="cg",
        help="result tree: results_cg or results_aa (or both)",
    )
    parser.add_argument(
        "--case",
        choices=list(CASE_LABELS) + ["all"],
        default="all",
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    cases = list(CASE_LABELS) if args.case == "all" else [args.case]
    systems = list(SYSTEMS) if args.system == "all" else [args.system]
    for system in systems:
        collect_system(system, cases, plot=args.plot)


if __name__ == "__main__":
    main()

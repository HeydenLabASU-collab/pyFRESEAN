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
    BENCH_T_VDOS,
    BENCH_T_EIGEN,
    BENCH_T_FRAME_PROCESSING,
    BENCH_T_MAPPING,
    BENCH_T_PY_COARSE,
    BENCH_T_PY_FRESEAN,
    BENCH_T_PY_TOTAL,
    BENCH_T_CG_TOTAL,
    BENCH_T_FRESEAN_TOTAL,
    BENCH_T_READ_TRAJ,
    BENCH_T_VELOCITY_SPECTRA,
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
    "omp1_njobs_n_vec": "OMP=1, n_jobs=N, n_workers=1 (vectorized tiles)",
    "omp_hybrid_vec": "hybrid: corr n_jobs=N, rfft/eigh OMP=N (vectorized)",
    "omp_n_njobs_1_vec_old": "OMP=N, n_jobs=1, n_workers=1 (vectorized, pre-rfft)",
}

VECTORIZED_COMPARE_CASES = (
    "omp_n_njobs_1_vec_old",
    "omp_n_njobs_1_vec",
    "omp1_njobs_n_vec",
    "omp_hybrid_vec",
)

CASE_ORDER = (
    "omp1_njobs_n_nworkers_1",
    "omp1_njobs_n_nworkers_2",
    "omp1_njobs_n_nworkers_n",
    "omp_n_njobs_1",
    "omp_n_njobs_1_vec",
    "omp1_njobs_n_vec",
    "omp_hybrid_vec",
    "omp_n_njobs_1_vec_old",
)


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


def py_fresean_velocity_matrix_time(row: dict) -> float:
    """Trajectory read + velocity FFT (plotted as one phase)."""
    return nested_bench(row, "py_fresean", BENCH_T_READ_TRAJ) + nested_bench(
        row, "py_fresean", BENCH_T_VELOCITY_SPECTRA
    )


def py_fresean_corr_phase_time(row: dict) -> float:
    """Correlation build + VDOS normalization (plotted as one phase)."""
    return nested_bench(row, "py_fresean", BENCH_T_CORR_MATRIX) + nested_bench(
        row, "py_fresean", BENCH_T_VDOS
    )


def py_fresean_eigen_phase_time(row: dict) -> float:
    """Mode diagonalization (plotted as one phase)."""
    return nested_bench(row, "py_fresean", BENCH_T_EIGEN)


def py_fresean_time(row: dict) -> float:
    return bench_value(row, BENCH_T_PY_FRESEAN) or nested_bench(
        row, "py_fresean", BENCH_T_FRESEAN_TOTAL
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


def _baseline_time(rows: list[dict]) -> float | None:
    for row in rows:
        if int(row["ncpus"]) == 1:
            return py_fresean_time(row)
    return None


def _c_baseline_time(c_rows: list[dict]) -> float | None:
    for row in c_rows:
        if int(row["ncpus"]) == 1:
            return c_spectral_time(row)
    return None


def _inverse_walltime(walltime: float) -> float:
    if walltime <= 0.0:
        return 0.0
    return 1.0 / walltime


def _speedup(baseline: float, walltime: float) -> float:
    if walltime <= 0.0:
        return 0.0
    return baseline / walltime


def discover_cases(py_cases_dir: Path) -> list[str]:
    """Known cases with results, then any extra dirs, in stable plot order."""
    order = {name: index for index, name in enumerate(CASE_ORDER)}
    known = [
        name
        for name in CASE_ORDER
        if name in CASE_LABELS and (py_cases_dir / name).is_dir()
    ]
    extra = sorted(
        p.name
        for p in py_cases_dir.iterdir()
        if p.is_dir()
        and p.name not in CASE_LABELS
        and any(p.glob("ncpus_*/result.json"))
    )
    return sorted(
        known + extra, key=lambda name: order.get(name, len(CASE_ORDER))
    )


def _plot_metric_overview(
    case_rows: dict[str, list[dict]],
    c_rows: list[dict],
    plot_path: Path,
    *,
    system: str,
    title: str,
    ylabel: str,
    metric_fn,
    c_metric_fn=None,
    reference_line: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ncpus_ref: list[int] | None = None
    c_map = {int(r["ncpus"]): r for r in c_rows}
    c_baseline = _c_baseline_time(c_rows) if c_rows else None

    for case, rows in case_rows.items():
        if not rows:
            continue
        baseline = _baseline_time(rows)
        if baseline is None:
            continue
        ncpus = [int(r["ncpus"]) for r in rows]
        ncpus_ref = ncpus
        values = [
            metric_fn(baseline, n, py_fresean_time(r))
            for r, n in zip(rows, ncpus)
        ]
        ax.plot(ncpus, values, "o-", label=CASE_LABELS.get(case, case))

    if (
        ncpus_ref
        and c_map
        and c_baseline is not None
        and c_metric_fn is not None
    ):
        c_ncpus = [n for n in ncpus_ref if n in c_map]
        if c_ncpus:
            ax.plot(
                c_ncpus,
                [
                    c_metric_fn(
                        c_baseline,
                        n,
                        c_spectral_time(c_map[n]),
                    )
                    for n in c_ncpus
                ],
                "s--",
                color="black",
                linewidth=2,
                label="C covar+eigen",
            )

    if reference_line is not None:
        ax.axhline(reference_line, color="gray", linestyle=":", linewidth=1.2)

    ax.set_xlabel("CPUs")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _setup_log_axis(ax, ncpus_ref or list(CPU_COUNTS))
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


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


def plot_inverse_walltime_overview(
    case_rows: dict[str, list[dict]],
    c_rows: list[dict],
    plot_path: Path,
    *,
    system: str,
) -> None:
    _plot_metric_overview(
        case_rows,
        c_rows,
        plot_path,
        system=system,
        title=f"HEWL {system.upper()} throughput — 1 / wall time",
        ylabel=r"$1 / T_N$ (s$^{-1}$)",
        metric_fn=lambda _baseline, _ncpus, walltime: _inverse_walltime(
            walltime
        ),
        c_metric_fn=lambda _baseline, _ncpus, walltime: _inverse_walltime(
            walltime
        ),
    )


def plot_speedup_overview(
    case_rows: dict[str, list[dict]],
    c_rows: list[dict],
    plot_path: Path,
    *,
    system: str,
) -> None:
    _plot_metric_overview(
        case_rows,
        c_rows,
        plot_path,
        system=system,
        title=f"HEWL {system.upper()} speedup — single-core time / wall time",
        ylabel=r"$T_1 / T_N$",
        metric_fn=lambda baseline, _ncpus, walltime: _speedup(
            baseline, walltime
        ),
        c_metric_fn=lambda baseline, _ncpus, walltime: _speedup(
            baseline, walltime
        ),
        reference_line=1.0,
    )


def plot_vectorized_comparison(
    case_rows: dict[str, list[dict]],
    plot_path: Path,
    *,
    system: str,
) -> None:
    """Compare pre-vectorized, vectorized serial, and vectorized parallel runs."""
    import matplotlib.pyplot as plt

    series: list[tuple[str, dict[int, dict], float | None]] = []
    for case in VECTORIZED_COMPARE_CASES:
        rows = case_rows.get(case) or []
        if not rows:
            continue
        series.append(
            (
                CASE_LABELS.get(case, case),
                {int(r["ncpus"]): r for r in rows},
                _baseline_time(rows),
            )
        )
    if len(series) < 2:
        return

    ncpus = sorted(
        set.intersection(*(set(data.keys()) for _, data, _ in series))
    )
    if not ncpus:
        return

    markers = ("o-", "s-", "^-", "D-")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = (
        ("wall time (s)", lambda r: py_fresean_time(r)),
        (
            r"$1 / T_N$ (s$^{-1}$)",
            lambda r: _inverse_walltime(py_fresean_time(r)),
        ),
        (r"$T_1 / T_N$", None),
    )

    for ax, (ylabel, value_fn) in zip(axes, metrics):
        if value_fn is None:
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2)
        for index, (label, data, baseline) in enumerate(series):
            if value_fn is None:
                values = [
                    _speedup(baseline or 0.0, py_fresean_time(data[n]))
                    for n in ncpus
                ]
            else:
                values = [value_fn(data[n]) for n in ncpus]
            ax.plot(
                ncpus,
                values,
                markers[index % len(markers)],
                label=label,
            )
        ax.set_xlabel("CPUs")
        ax.set_ylabel(ylabel)
        _setup_log_axis(ax, ncpus)
        ax.legend(fontsize=7)

    axes[0].set_title(f"HEWL {system.upper()} — corr matrix implementations")
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


def plot_case_speedup(
    case: str,
    rows: list[dict],
    c_rows: list[dict],
    plot_path: Path,
    *,
    system: str,
) -> None:
    import matplotlib.pyplot as plt

    baseline = _baseline_time(rows)
    if baseline is None:
        return
    ncpus = [int(r["ncpus"]) for r in rows]
    c_map = {int(r["ncpus"]): r for r in c_rows}
    c_baseline = _c_baseline_time(c_rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        ncpus,
        [_speedup(baseline, py_fresean_time(r)) for r in rows],
        "o-",
        label="pyfresean FRESEAN",
    )
    if c_map and c_baseline is not None:
        ax.plot(
            ncpus,
            [
                _speedup(c_baseline, c_spectral_time(c_map.get(n)))
                for n in ncpus
            ],
            "s-",
            label="C covar+eigen",
        )
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xlabel("CPUs")
    ax.set_ylabel(r"$T_1 / T_N$")
    ax.set_title(
        f"HEWL {system.upper()} speedup — {CASE_LABELS.get(case, case)}"
    )
    _setup_log_axis(ax, ncpus)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def plot_case_inverse_walltime(
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
        [_inverse_walltime(py_fresean_time(r)) for r in rows],
        "o-",
        label="pyfresean FRESEAN",
    )
    if c_map:
        ax.plot(
            ncpus,
            [_inverse_walltime(c_spectral_time(c_map.get(n))) for n in ncpus],
            "s-",
            label="C covar+eigen",
        )
    ax.set_xlabel("CPUs")
    ax.set_ylabel(r"$1 / T_N$ (s$^{-1}$)")
    ax.set_title(
        f"HEWL {system.upper()} throughput — {CASE_LABELS.get(case, case)}"
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
        [py_fresean_velocity_matrix_time(r) for r in rows],
        "o-",
        label="velocity matrix (read + FFT)",
    )
    ax.plot(
        ncpus,
        [py_fresean_corr_phase_time(r) for r in rows],
        "s-",
        label="corr matrix + vdos",
    )
    ax.plot(
        ncpus,
        [py_fresean_eigen_phase_time(r) for r in rows],
        "v-",
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
    ax.set_title(f"py CG reference (bench_t_cg_total={sum(values):.1f}s)")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def print_table(
    system: str, case: str, rows: list[dict], c_rows: list[dict]
) -> None:
    print(f"\n=== {system.upper()} — {CASE_LABELS.get(case, case)} ===")
    print(
        f"{'ncpus':>5}  {'py_fresean':>10}  {'speedup':>8}  "
        f"{'1/T':>10}  {'c_total':>10}"
    )
    c_map = {int(r["ncpus"]): r for r in c_rows}
    baseline = _baseline_time(rows)
    for r in rows:
        n = int(r["ncpus"])
        py_f = py_fresean_time(r)
        speedup = _speedup(baseline or 0.0, py_f) if baseline else 0.0
        inv_t = _inverse_walltime(py_f)
        c_total = c_spectral_time(c_map.get(n))
        print(
            f"{n:5d}  {py_f:10.2f}  {speedup:8.2f}  "
            f"{inv_t:10.4f}  {c_total:10.2f}"
        )


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
            case_plot_dir = plots / case
            plot_case_total(
                case,
                rows,
                c_rows,
                case_plot_dir / "total_vs_c.png",
                system=system,
            )
            plot_case_breakdown(
                case,
                rows,
                case_plot_dir / "breakdown.png",
                system=system,
            )
            plot_case_speedup(
                case,
                rows,
                c_rows,
                case_plot_dir / "speedup.png",
                system=system,
            )
            plot_case_inverse_walltime(
                case,
                rows,
                c_rows,
                case_plot_dir / "inverse_walltime.png",
                system=system,
            )
            print(f"Wrote {case_plot_dir}")

    if plot and case_rows:
        plot_total_overview(
            case_rows,
            c_rows,
            plots / "total_all_cases.png",
            system=system,
        )
        plot_inverse_walltime_overview(
            case_rows,
            c_rows,
            plots / "inverse_walltime_all_cases.png",
            system=system,
        )
        plot_speedup_overview(
            case_rows,
            c_rows,
            plots / "speedup_all_cases.png",
            system=system,
        )
        plot_vectorized_comparison(
            case_rows,
            plots / "vectorized_corr_comparison.png",
            system=system,
        )
        if system == "cg":
            plot_cg_reference(plots / "cg_reference_breakdown.png")
        print(f"Wrote {plots / 'total_all_cases.png'}")
        print(f"Wrote {plots / 'inverse_walltime_all_cases.png'}")
        print(f"Wrote {plots / 'speedup_all_cases.png'}")
        compare_count = sum(
            1 for case in VECTORIZED_COMPARE_CASES if case_rows.get(case)
        )
        if compare_count >= 2:
            print(f"Wrote {plots / 'vectorized_corr_comparison.png'}")

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

    systems = list(SYSTEMS) if args.system == "all" else [args.system]
    for system in systems:
        if args.case == "all":
            cases = discover_cases(py_cases_dir(system))
            if not cases:
                cases = list(CASE_LABELS)
        else:
            cases = [args.case]
        collect_system(system, cases, plot=args.plot)


if __name__ == "__main__":
    main()

"""Shared wall-time keys for pyfresean benchmarks (values are seconds)."""

from __future__ import annotations

# Coarse-grain phases
BENCH_T_MAPPING = "bench_t_mapping"
BENCH_T_FRAME_PROCESSING = "bench_t_frame_processing"
BENCH_T_WRITE_OUTPUTS = "bench_t_write_outputs"
BENCH_T_ASSEMBLE_UNIVERSE = "bench_t_assemble_universe"
BENCH_T_CG_TOTAL = "bench_t_cg_total"

CG_BENCHMARK_KEYS = (
    BENCH_T_MAPPING,
    BENCH_T_FRAME_PROCESSING,
    BENCH_T_WRITE_OUTPUTS,
    BENCH_T_ASSEMBLE_UNIVERSE,
    BENCH_T_CG_TOTAL,
)

# FRESEAN phases: read_traj → velocity_spectra → corr_matrix → vdos (norm) → eigen
BENCH_T_READ_TRAJ = "bench_t_read_traj"
BENCH_T_VELOCITY_SPECTRA = "bench_t_velocity_spectra"
BENCH_T_CORR_MATRIX = "bench_t_corr_matrix"
BENCH_T_VDOS = "bench_t_vdos"
BENCH_T_EIGEN = "bench_t_eigen"
BENCH_T_FRESEAN_TOTAL = "bench_t_fresean_total"

FRESEAN_BENCHMARK_KEYS = (
    BENCH_T_READ_TRAJ,
    BENCH_T_VELOCITY_SPECTRA,
    BENCH_T_CORR_MATRIX,
    BENCH_T_VDOS,
    BENCH_T_EIGEN,
    BENCH_T_FRESEAN_TOTAL,
)

# HEWL / external benchmark JSON totals
BENCH_T_PY_COARSE = "bench_t_py_coarse"
BENCH_T_PY_FRESEAN = "bench_t_py_fresean"
BENCH_T_PY_TOTAL = "bench_t_py_total"
BENCH_T_C_COARSE = "bench_t_c_coarse"
BENCH_T_C_COVAR = "bench_t_c_covar"
BENCH_T_C_EIGEN = "bench_t_c_eigen"
BENCH_T_C_TOTAL = "bench_t_c_total"


def new_cg_benchmark_timings() -> dict[str, float]:
    return {
        BENCH_T_MAPPING: 0.0,
        BENCH_T_FRAME_PROCESSING: 0.0,
        BENCH_T_WRITE_OUTPUTS: 0.0,
        BENCH_T_ASSEMBLE_UNIVERSE: 0.0,
    }


def new_fresean_benchmark_timings() -> dict[str, float]:
    return {
        key: 0.0
        for key in FRESEAN_BENCHMARK_KEYS
        if key != BENCH_T_FRESEAN_TOTAL
    }


def finalize_cg_benchmark(timings: dict[str, float]) -> dict[str, float]:
    for key in CG_BENCHMARK_KEYS:
        if key != BENCH_T_CG_TOTAL:
            timings.setdefault(key, 0.0)
    timings[BENCH_T_CG_TOTAL] = (
        timings[BENCH_T_MAPPING]
        + timings[BENCH_T_FRAME_PROCESSING]
        + timings[BENCH_T_WRITE_OUTPUTS]
        + timings[BENCH_T_ASSEMBLE_UNIVERSE]
    )
    return timings


def finalize_fresean_benchmark(timings: dict[str, float]) -> dict[str, float]:
    for key in FRESEAN_BENCHMARK_KEYS:
        if key != BENCH_T_FRESEAN_TOTAL:
            timings.setdefault(key, 0.0)
    timings[BENCH_T_FRESEAN_TOTAL] = (
        timings[BENCH_T_READ_TRAJ]
        + timings[BENCH_T_VELOCITY_SPECTRA]
        + timings[BENCH_T_CORR_MATRIX]
        + timings[BENCH_T_VDOS]
        + timings[BENCH_T_EIGEN]
    )
    return timings

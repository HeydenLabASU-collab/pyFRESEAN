"""Shared wall-time keys for pyfresean benchmarks (values are seconds)."""

from __future__ import annotations

# Coarse-grain phases
BENCH_T_MAPPING = "bench_t_mapping"
BENCH_T_FRAME_PROCESSING = "bench_t_frame_processing"
BENCH_T_WRITE_OUTPUTS = "bench_t_write_outputs"
BENCH_T_ASSEMBLE_UNIVERSE = "bench_t_assemble_universe"
BENCH_T_TOTAL = "bench_t_total"

CG_BENCHMARK_KEYS = (
    BENCH_T_MAPPING,
    BENCH_T_FRAME_PROCESSING,
    BENCH_T_WRITE_OUTPUTS,
    BENCH_T_ASSEMBLE_UNIVERSE,
    BENCH_T_TOTAL,
)

# FRESEAN phases
BENCH_T_VELOCITY_MATRIX = "bench_t_velocity_matrix"
BENCH_T_CORR_MATRIX = "bench_t_corr_matrix"
BENCH_T_EIGEN = "bench_t_eigen"
BENCH_T_SPECTRAL = "bench_t_spectral"

FRESEAN_BENCHMARK_KEYS = (
    BENCH_T_VELOCITY_MATRIX,
    BENCH_T_CORR_MATRIX,
    BENCH_T_EIGEN,
    BENCH_T_SPECTRAL,
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


def finalize_cg_benchmark(timings: dict[str, float]) -> dict[str, float]:
    timings[BENCH_T_TOTAL] = (
        timings[BENCH_T_MAPPING]
        + timings[BENCH_T_FRAME_PROCESSING]
        + timings[BENCH_T_WRITE_OUTPUTS]
        + timings[BENCH_T_ASSEMBLE_UNIVERSE]
    )
    return timings


def finalize_fresean_benchmark(timings: dict[str, float]) -> dict[str, float]:
    timings[BENCH_T_SPECTRAL] = (
        timings[BENCH_T_VELOCITY_MATRIX]
        + timings[BENCH_T_CORR_MATRIX]
        + timings[BENCH_T_EIGEN]
    )
    return timings

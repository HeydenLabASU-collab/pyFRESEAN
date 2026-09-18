"""Tests for per-phase BLAS/thread-pool helpers."""

from __future__ import annotations

import os

import numpy as np
import pytest
import threadpoolctl

from pyfresean.parallel import (
    BLAS_THREAD_ENV_VARS,
    PhaseParallelism,
    blas_thread_context,
    build_phase_parallelism,
    set_blas_threads,
)


def test_build_phase_parallelism_defaults():
    phases = build_phase_parallelism()
    assert phases["velocity_fft"] == PhaseParallelism()
    assert phases["corr_matrix"] == PhaseParallelism()
    assert phases["eigen"] == PhaseParallelism()


def test_build_phase_parallelism_per_phase():
    phases = build_phase_parallelism(
        parallel={
            "corr_matrix": {"n_jobs": 4, "omp_threads": 1},
            "eigen": {"n_jobs": 1, "omp_threads": 8},
        }
    )
    assert phases["corr_matrix"] == PhaseParallelism(n_jobs=4, omp_threads=1)
    assert phases["eigen"] == PhaseParallelism(n_jobs=1, omp_threads=8)
    assert phases["velocity_fft"] == PhaseParallelism()


def test_build_phase_parallelism_unknown_phase():
    with pytest.raises(ValueError, match="unknown parallel phase"):
        build_phase_parallelism(parallel={"bad": {"n_jobs": 1}})


def test_blas_thread_context_restores_env():
    os.environ["OMP_NUM_THREADS"] = "7"
    with blas_thread_context(3):
        assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["OMP_NUM_THREADS"] == "7"


def test_blas_thread_context_applies_runtime_limit():
    matrix = np.random.default_rng(0).standard_normal((64, 64))
    np.linalg.eigh(matrix)

    with blas_thread_context(2):
        infos = threadpoolctl.threadpool_info()
        if not infos:
            pytest.skip("no thread pools registered after eigh")
        for info in infos:
            assert info["num_threads"] <= 2


def test_set_blas_threads_updates_env():
    saved = {var: os.environ.get(var) for var in BLAS_THREAD_ENV_VARS}
    try:
        set_blas_threads(5)
        for var in BLAS_THREAD_ENV_VARS:
            assert os.environ[var] == "5"
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value

"""Per-phase BLAS and thread-pool settings for FRESEAN :meth:`_conclude`."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping, Optional

import threadpoolctl

FRESEAN_PHASES = ("velocity_fft", "corr_matrix", "eigen")

BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class PhaseParallelism:
    """Thread-pool and BLAS settings for one :meth:`~pyfresean.FRESEAN._conclude` phase."""

    n_jobs: int = 1
    omp_threads: int = 1


def set_blas_threads(omp_threads: int) -> None:
    """Set BLAS/OpenMP thread env vars seen by NumPy and SciPy."""
    n = max(1, int(omp_threads))
    for var in BLAS_THREAD_ENV_VARS:
        os.environ[var] = str(n)


@contextmanager
def blas_thread_context(omp_threads: int) -> Iterator[None]:
    """Temporarily limit BLAS/OpenMP threads for one :meth:`~pyfresean.FRESEAN._conclude` phase.

    Env vars are updated for libraries that read them at import time. ``threadpoolctl``
    applies runtime limits so thread counts can change safely between phases (e.g. corr
    thread pool at ``omp_threads=1`` followed by eigen at ``omp_threads=N``).
    """
    n = max(1, int(omp_threads))
    saved = {var: os.environ.get(var) for var in BLAS_THREAD_ENV_VARS}
    set_blas_threads(n)
    try:
        with threadpoolctl.threadpool_limits(limits=n):
            yield
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


def resolve_n_jobs(n_jobs: int) -> int:
    """Resolve ``1``, ``-1`` (all CPUs), or an explicit worker count."""
    if n_jobs == 1:
        return 1
    if n_jobs < 0:
        return os.cpu_count() or 1
    return n_jobs


def _validate_n_jobs(n_jobs: int, *, label: str) -> None:
    if n_jobs != 1 and n_jobs != -1 and n_jobs < 2:
        raise ValueError(
            f"{label} n_jobs must be 1, -1, or an integer >= 2; got {n_jobs!r}"
        )


def _parse_phase_values(
    phase: str,
    values: Mapping[str, int],
    base: PhaseParallelism,
) -> PhaseParallelism:
    n_jobs = int(values.get("n_jobs", base.n_jobs))
    omp_threads = int(values.get("omp_threads", base.omp_threads))
    if omp_threads < 1:
        raise ValueError(
            f"parallel[{phase!r}]['omp_threads'] must be >= 1; got {omp_threads!r}"
        )
    _validate_n_jobs(n_jobs, label=f"parallel[{phase!r}]")
    return PhaseParallelism(n_jobs=n_jobs, omp_threads=omp_threads)


def build_phase_parallelism(
    parallel: Optional[Mapping[str, Mapping[str, int]]] = None,
) -> dict[str, PhaseParallelism]:
    """Build per-phase settings with defaults of ``n_jobs=1``, ``omp_threads=1``.

    Parameters
    ----------
    parallel
        Optional dict keyed by phase name (``velocity_fft``, ``corr_matrix``,
        ``eigen``). Each value may contain ``n_jobs`` and/or ``omp_threads``.
    """
    phases = {name: PhaseParallelism() for name in FRESEAN_PHASES}

    if parallel is not None:
        for phase, values in parallel.items():
            if phase not in phases:
                raise ValueError(
                    f"unknown parallel phase {phase!r}; "
                    f"choose from: {', '.join(FRESEAN_PHASES)}"
                )
            phases[phase] = _parse_phase_values(phase, values, phases[phase])

    return phases

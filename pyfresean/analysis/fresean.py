"""
FRESEAN --- :mod:`pyfresean.analysis.FRESEAN`
===========================================================

This module contains the :class:`FRESEAN` class.

"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import numpy as np
from MDAnalysis.analysis.backends import BackendBase, BackendSerial

LagSymmetrization = Literal["mirror", "average"]

from MDAnalysis.analysis.base import AnalysisBase, Results
from MDAnalysis.analysis.results import ResultsGroup
from scipy.fft import ifft, irfft, rfft

from pyfresean.benchmark_keys import (
    BENCH_T_CORR_MATRIX,
    BENCH_T_EIGEN,
    BENCH_T_READ_TRAJ,
    BENCH_T_VDOS,
    BENCH_T_VELOCITY_SPECTRA,
    finalize_fresean_benchmark,
    new_fresean_benchmark_timings,
)
from pyfresean.parallel import (
    PhaseParallelism,
    blas_thread_context,
    build_phase_parallelism,
    resolve_n_jobs,
)

if TYPE_CHECKING:
    from MDAnalysis.core.groups import AtomGroup
    from MDAnalysis.core.universe import Universe

ComputeOption = Union[bool, Sequence[float]]


@dataclass(frozen=True)
class _FreseanRunPlan:
    """Internal work list for one :meth:`FRESEAN.run` (from :meth:`_fresean_run_plan`)."""

    calc_read_traj: Optional[bool]
    calc_velocity_fft: Optional[bool]
    calc_vdos: Optional[bool]
    calc_full_corr_matrix: bool
    calc_autocorr_diagonal: Optional[bool]
    calc_modes: bool


class FRESEAN(AnalysisBase):
    """FRESEAN class.

    This class is used to perform FRESEAN velocity-correlation analysis on a
    trajectory.

    Parameters
    ----------
    universe_or_atomgroup: Universe or AtomGroup
        Universe or group of atoms to apply this analysis to.
        If a trajectory is associated with the atoms,
        then the computation iterates over the trajectory.
    select: str
        Selection string for atoms to extract from the input Universe or
        AtomGroup
    n_constraints: int
        Number of constraints applied to atoms in the selection
    n_corr: int
        Number of correlation time frames
    dt: float
        Time step in the trajectory, in ps
    sigma: float
        Width of the Gaussian window function, in cm**-1
    lag_symmetrization: "mirror" or "average"
        How to enforce time symmetry of the correlation function after the
        first inverse FFT. ``"mirror"`` (default) matches the FRESEAN_tutorial
        notebooks: positive lags are taken from ``ifft`` indices ``0 … n_corr-1``
        and the negative-lag half for the windowed FFT is built by reversing
        those values. ``"average"`` matches ``gen-modes_omp.c``: for each lag
        ``k > 0``, average ``ifft[k]`` with ``ifft[n_frames - k]`` (the wrapped
        negative-lag bin from :func:`scipy.fft.ifft`), then mirror for the
        second half. The latter is more appropriate for cross-correlations on
        finite trajectories.
    parallel: dict or None
        Optional per-phase threading for :meth:`_conclude`. Keys are
        ``velocity_fft``, ``corr_matrix``, and ``eigen``. Each value is a dict
        with optional ``n_jobs`` (``ThreadPoolExecutor`` workers, default
        ``1``) and ``omp_threads`` (BLAS/OpenMP threads inside NumPy/SciPy,
        default ``1``). Example — tile pool for corr, BLAS for eigen::

            parallel={
                "velocity_fft": {"n_jobs": 1, "omp_threads": 8},
                "corr_matrix": {"n_jobs": 8, "omp_threads": 1},
                "eigen": {"n_jobs": 1, "omp_threads": 8},
            }
    Spectral outputs are configured per :meth:`run` (not at construction).
    See :meth:`run` for ``read_traj``, ``compute_vdos``, ``compute_corr_matrix``,
    and ``compute_modes``.

    run(..., read_traj=True, compute_vdos=True, compute_corr_matrix=True,
        compute_modes=True, ...)
        Pipeline: trajectory (optional) → velocity FFT → normalized
        correlation matrix → VDOS and/or modes. Enabling ``compute_vdos``
        builds the correlation matrix and fills ``results.vdos_total``.
        ``compute_modes`` diagonalizes that matrix; if ``results.corr_matrix``
        is already present for the same frame selection, it is reused.
        ``compute_vdos`` / ``compute_modes`` are each ``True`` (all bins),
        ``False`` (skip), or a wavenumber list (nearest grid bin). Intermediate
        arrays live on :attr:`results` (``velocity_spectra``, ``corr_matrix``).

        When ``n_workers`` > 1 and a parallel backend is used, MDAnalysis
        splits the trajectory across workers for :meth:`_single_frame`
        velocity collection. :meth:`_conclude` still runs once on the parent
        process after merging partial results. Requires parallelizable
        trajectory transformations (e.g. :class:`pyfresean.Align`).

    Attributes
    ----------
    universe: :class:`~MDAnalysis.core.universe.Universe`
        The universe to which this analysis is applied
    atomgroup: :class:`~MDAnalysis.core.groups.AtomGroup`
        The atoms to which this analysis is applied
    results: :class:`~MDAnalysis.analysis.base.Results`
        Populated by :meth:`run`. With ``read_traj=True``,
        ``compute_vdos=True``, and ``compute_modes=True`` (defaults), after
        one full run you get:

        * ``freqs``, ``win_time``, ``n_dof`` — analysis grid and window
        * ``velocity_spectra`` — complex RFFT of mass-weighted velocities
        * ``corr_matrix``, ``corr_freqs`` — normalized ``(n_corr, n, n)``
        * ``avg_temperature``, ``vdos_norm``
        * ``vdos_freqs``, ``vdos_total``
        * ``mode_freqs``, ``eigenvalues``, ``eigenvectors``
        * ``fresean_frame_key`` — frame selection tag for ``read_traj=False``

        Time-domain ``velocities`` are not kept after :meth:`_conclude`.
    start: Optional[int]
        The first frame of the trajectory used to compute the analysis
    stop: Optional[int]
        The frame to stop at for the analysis
    step: Optional[int]
        Number of frames to skip between each analyzed frame
    n_frames: int
        Number of frames analysed in the trajectory
    times: numpy.ndarray
        array of Timestep times. Only exists after calling
        :meth:`FRESEAN.run`
    frames: numpy.ndarray
        array of Timestep frame indices. Only exists after calling
        :meth:`FRESEAN.run`
    """

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        return ("serial", "multiprocessing", "dask")

    @staticmethod
    def _take_first_result(values):
        return values[0]

    def _fresean_cache_check(
        self,
        frame_key: tuple[Any, ...],
        n_axes: int,
        corr_shape: tuple[int, int, int],
    ) -> tuple[bool, bool, bool, bool]:
        """Whether prior :attr:`results` on this instance can be reused for ``frame_key``.

        Inspects ``self.results`` only (not worker merge / :meth:`_get_aggregator`).
        """
        res = getattr(self, "results", None)
        frame_ok = (
            res is not None
            and getattr(res, "fresean_frame_key", None) == frame_key
        )
        sp = getattr(res, "velocity_spectra", None) if res else None
        has_spectra = (
            frame_ok
            and isinstance(sp, np.ndarray)
            and sp.shape == (n_axes, sp.shape[1])
            and sp.size > 0
            and np.any(sp)
        )
        has_velocities = (
            frame_ok
            and hasattr(res, "velocities")
            and isinstance(res.velocities, np.ndarray)
            and res.velocities.size > 0
            and np.any(res.velocities)
        )
        cm = getattr(res, "corr_matrix", None) if res else None
        has_corr = (
            frame_ok
            and isinstance(cm, np.ndarray)
            and cm.shape == corr_shape
            and np.any(cm)
        )
        has_vdos_norm = frame_ok and getattr(res, "vdos_norm", None) is not None
        return has_spectra, has_velocities, has_corr, has_vdos_norm

    def _fresean_run_plan(
        self,
        read_traj: bool,
        compute_vdos: bool,
        compute_corr_matrix: bool,
        compute_modes: bool,
        frame_key: tuple[Any, ...],
        n_axes: int,
        corr_shape: tuple[int, int, int],
    ) -> _FreseanRunPlan:
        """Work backwards from requested outputs to trajectory / spectral steps."""
        calc_modes = compute_modes
        calc_vdos = compute_vdos
        calc_full_corr = compute_corr_matrix
        calc_autocorr = None
        calc_read_traj = read_traj
        calc_velocity_fft = None

        has_spectra, has_velocities, has_corr, has_vdos_norm = (
            self._fresean_cache_check(frame_key, n_axes, corr_shape)
        )

        if calc_modes:
            if not calc_full_corr and (not has_corr or not has_vdos_norm):
                warnings.warn(
                    "compute_modes requires corr_matrix and vdos_norm in pre-computed in cache, but is not present, so running that calculation",
                    UserWarning,
                    stacklevel=2,
                )
                calc_full_corr = True
        if calc_full_corr or calc_vdos:
            if not calc_full_corr:
                calc_autocorr = True
            if not has_spectra:
                if has_velocities and not calc_read_traj:
                    warnings.warn(
                        "compute_corr_matrix requires velocity_spectra in pre-computed in cache, but is not present, computing the velocity FFT from available velocities",
                        UserWarning,
                        stacklevel=2,
                    )
                    calc_velocity_fft = True
                elif not has_velocities and not calc_read_traj:
                    warnings.warn(
                        "compute_corr_matrix requires velocity_spectra in pre-computed in cache, but is not present, and no velocities are available, so reading the trajectory and computing the velocity FFT",
                        UserWarning,
                        stacklevel=2,
                    )
                    calc_read_traj = True
                    calc_velocity_fft = True
        if calc_read_traj:
            calc_velocity_fft = True

        return _FreseanRunPlan(
            calc_read_traj=calc_read_traj,
            calc_velocity_fft=calc_velocity_fft,
            calc_full_corr_matrix=calc_full_corr,
            calc_autocorr_diagonal=calc_autocorr,
            calc_vdos=calc_vdos,
            calc_modes=calc_modes,
        )

    def _reset_fresean_benchmark_phases(
        self, timings: dict[str, float], plan: _FreseanRunPlan
    ) -> None:
        """Zero benchmark keys for phases that will run this ``run()`` call."""
        if plan.calc_read_traj is True:
            timings[BENCH_T_READ_TRAJ] = 0.0

        if plan.calc_velocity_fft is True:
            timings[BENCH_T_VELOCITY_SPECTRA] = 0.0

        if plan.calc_full_corr_matrix or plan.calc_autocorr_diagonal is True:
            timings[BENCH_T_CORR_MATRIX] = 0.0
            timings[BENCH_T_VDOS] = 0.0

        if plan.calc_modes:
            timings[BENCH_T_EIGEN] = 0.0

    @staticmethod
    def _split_compute_option(
        option: ComputeOption, name: str
    ) -> tuple[bool, Optional[Sequence[float]]]:
        if isinstance(option, (bool, np.bool_)):
            return bool(option), None
        if isinstance(option, (list, tuple, np.ndarray)):
            if len(option) == 0:
                raise ValueError(f"{name} frequency list must not be empty")
            return True, list(option)
        raise TypeError(
            f"{name} must be True, False, or a sequence of wavenumbers (cm**-1)"
        )

    def _get_aggregator(self):
        lookup: dict[str, Any] = {
            "velocities": ResultsGroup.ndarray_hstack,
            "freqs": FRESEAN._take_first_result,
            "win_time": FRESEAN._take_first_result,
            "n_dof": FRESEAN._take_first_result,
        }
        return ResultsGroup(lookup=lookup)

    def __init__(
        self,
        universe_or_atomgroup: Union["Universe", "AtomGroup"],
        select: str = "all",
        n_constraints: int = 0,
        n_corr: int = 500,
        dt: float = 0.004,
        sigma: float = 10.0,
        lag_symmetrization: LagSymmetrization = "mirror",
        parallel: Optional[Mapping[str, Mapping[str, int]]] = None,
        **kwargs,
    ):
        # the below line must be kept to initialize the AnalysisBase class!
        super().__init__(universe_or_atomgroup.trajectory, **kwargs)
        # after this you will be able to access `self.results`
        # `self.results` is a dictionary-like object
        # that can should used to store and retrieve results
        # See more at the MDAnalysis documentation:
        # https://docs.mdanalysis.org/stable/documentation_pages/analysis/base.html?highlight=results#MDAnalysis.analysis.base.Results

        self.universe = universe_or_atomgroup.universe
        self.atomgroup = universe_or_atomgroup.select_atoms(select)
        self.n_constraints = n_constraints
        self.n_corr = n_corr
        self.dt = dt
        self.sigma = sigma
        if lag_symmetrization not in ("mirror", "average"):
            raise ValueError(
                "lag_symmetrization must be 'mirror' or 'average', "
                f"got {lag_symmetrization!r}"
            )
        self.lag_symmetrization = lag_symmetrization
        self._parallel = build_phase_parallelism(parallel)
        self.benchmark: Optional[dict[str, float]] = None
        self._run_plan: Optional[_FreseanRunPlan] = None
        self._mode_freqs_req: Optional[Sequence[float]] = None
        self._vdos_freqs_req: Optional[Sequence[float]] = None

    @staticmethod
    def _frequency_grid(n_corr: int, dt: float) -> np.ndarray:
        wn0 = 1.0 / ((2 * n_corr - 1) * dt) * 33.3564
        return np.arange(n_corr) * wn0

    def _update_freq_indices(self) -> None:
        grid = self._frequency_grid(self.n_corr, self.dt)
        self._mode_freq_idx = (
            self._resolve_freq_indices(grid, self._mode_freqs_req)
            if self.compute_modes
            else np.array([], dtype=np.intp)
        )
        self._vdos_freq_idx = (
            self._resolve_freq_indices(grid, self._vdos_freqs_req)
            if self.compute_vdos
            else np.array([], dtype=np.intp)
        )

    @staticmethod
    def _resolve_freq_indices(
        grid: np.ndarray, requested: Optional[Sequence[float]]
    ) -> np.ndarray:
        if requested is None:
            return np.arange(grid.size, dtype=np.intp)
        indices = np.empty(len(requested), dtype=np.intp)
        for i, wn in enumerate(requested):
            indices[i] = int(np.argmin(np.abs(grid - wn)))
        return indices

    @staticmethod
    def _avg_temperature_from_traces(
        traces: np.ndarray,
        n_corr: int,
        win_time: np.ndarray,
        n_dof: int,
    ) -> float:
        return float(
            (traces[0] + 2.0 * np.sum(traces[1:]))
            / (2 * n_corr - 1)
            / win_time[0]
            / (8.3145 * 0.1)
            / n_dof
        )

    def _frame_key(
        self,
        start: Optional[int],
        stop: Optional[int],
        step: Optional[int],
        frames: Optional[Any],
    ) -> tuple[Any, ...]:
        if frames is not None:
            frame_key = tuple(frames)
        else:
            frame_key = (start, stop, step)
        return (self.n_corr, self.dt, self.sigma, frame_key)

    def _phase_parallel(self, phase: str) -> PhaseParallelism:
        return self._parallel[phase]

    @staticmethod
    def _add_benchmark_time(
        timings: Optional[dict[str, float]], key: str, elapsed: float
    ) -> None:
        if timings is not None:
            timings[key] = timings.get(key, 0.0) + elapsed

    def _benchmark_timings_for_run(self) -> dict[str, float]:
        """Phase timings for this run: fresh zeros, or prior :attr:`benchmark`."""
        if self.benchmark is None:
            return new_fresean_benchmark_timings()
        timings = dict(self.benchmark)
        for key in new_fresean_benchmark_timings():
            timings.setdefault(key, 0.0)
        return timings

    def _rfft_velocities(self, velocities: np.ndarray) -> np.ndarray:
        """Real FFT of the mass-weighted velocity matrix along the time axis."""
        spec = self._phase_parallel("velocity_fft")
        n_elements = velocities.shape[0]
        n_freq = velocities.shape[1] // 2 + 1
        spectra = np.empty((n_elements, n_freq), dtype=np.complex128)

        def transform_rows(i0: int, i1: int) -> None:
            spectra[i0:i1] = rfft(velocities[i0:i1], axis=1)

        with blas_thread_context(spec.omp_threads):
            max_workers = resolve_n_jobs(spec.n_jobs)
            if max_workers == 1:
                transform_rows(0, n_elements)
            else:
                row_chunks = np.array_split(np.arange(n_elements), max_workers)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    list(
                        executor.map(
                            lambda rows: transform_rows(
                                int(rows[0]), int(rows[-1]) + 1
                            ),
                            row_chunks,
                        )
                    )
        return spectra

    def _build_windowed_lags(
        self,
        tmp_time: np.ndarray,
        n_corr: int,
        n_frames: int,
    ) -> np.ndarray:
        """Pack symmetrized lags into the length ``2 * n_corr - 1`` FFT input."""
        return self._window_time_batch(
            tmp_time,
            n_corr,
            n_frames,
            win_time=None,
            lag_symmetrization=self.lag_symmetrization,
        )

    @staticmethod
    def _window_time_batch(
        tmp_time: np.ndarray,
        n_corr: int,
        n_frames: int,
        win_time: np.ndarray | None,
        lag_symmetrization: LagSymmetrization,
    ) -> np.ndarray:
        """Window and mirror/average lag blocks for arbitrary leading batch dims."""
        lag_len = 2 * n_corr - 1
        windowed = np.zeros(tmp_time.shape[:-1] + (lag_len,), dtype=np.float64)
        windowed[..., 0] = tmp_time[..., 0]
        if lag_symmetrization == "average" and n_corr > 1:
            k = np.arange(1, n_corr)
            windowed[..., 1:n_corr] = (
                tmp_time[..., 1:n_corr] + tmp_time[..., n_frames - k]
            ) / 2.0
        else:
            windowed[..., :n_corr] = tmp_time[..., :n_corr]
        windowed[..., n_corr:] = windowed[..., n_corr - 1 : 0 : -1]
        if win_time is not None:
            windowed *= win_time
        return windowed

    @staticmethod
    def _corr_matrix_budget_pairs(
        n_frames: int,
        max_working_bytes: int = 512 * 1024 * 1024,
        n_temp_arrays: int = 4,
    ) -> int:
        """Max ``(i, j)`` pairs per correlation-matrix tile within scratch RAM."""
        per_pair = max(n_frames * 8 * n_temp_arrays, 1)
        return max(1, max_working_bytes // per_pair)

    @staticmethod
    def _corr_matrix_tiles(
        n_elements: int,
        budget_pairs: int,
    ) -> list[tuple[int, int, int, int]]:
        """Upper-triangle tile bounds ``(i0, i1, j0, j1)`` for one corr build."""
        tiles: list[tuple[int, int, int, int]] = []
        i0 = 0
        while i0 < n_elements:
            rem_cols = n_elements - i0
            rem_rows = n_elements - i0
            ideal_row = budget_pairs / rem_cols
            if ideal_row >= 1.0:
                row_size = max(1, min(rem_rows, int(ideal_row)))
                tiles.append((i0, i0 + row_size, i0, n_elements))
                i0 += row_size
            else:
                col_size = max(1, min(rem_cols, int(budget_pairs)))
                j0 = i0
                while j0 < n_elements:
                    tiles.append(
                        (i0, i0 + 1, j0, min(j0 + col_size, n_elements))
                    )
                    j0 += col_size
                i0 += 1
        return tiles

    def _symmetrize_corr_matrix(
        self, corr_matrix: np.ndarray, n_elements: int
    ) -> None:
        i_upper, j_upper = np.triu_indices(n_elements, k=1)
        corr_matrix[:, j_upper, i_upper] = corr_matrix[:, i_upper, j_upper]

    def _fill_corr_matrix_tile_from_spectra(
        self,
        tile: tuple[int, int, int, int],
        spectra: np.ndarray,
        corr_matrix: np.ndarray,
        win_time: np.ndarray,
        n_corr: int,
        n_frames: int,
    ) -> None:
        """Fill one upper-triangle tile of ``corr_matrix`` from velocity spectra."""
        i0, i1, j0, j1 = tile
        row_sp = spectra[i0:i1]
        col_sp = spectra[j0:j1]
        cross_freq = np.real(row_sp[:, None, :] * np.conj(col_sp[None, :, :]))
        tmp_time = np.real(irfft(cross_freq, n=n_frames, axis=-1))
        windowed = self._window_time_batch(
            tmp_time,
            n_corr,
            n_frames,
            win_time,
            self.lag_symmetrization,
        )
        # windowed is real with length 2*n_corr-1, so rfft yields n_corr bins.
        corr_bins = np.real(rfft(windowed, axis=-1))
        corr_matrix[:, i0:i1, j0:j1] = np.moveaxis(corr_bins, -1, 0)

    def _build_corr_matrix(
        self,
        spectra: np.ndarray,
        corr_matrix: np.ndarray,
        win_time: np.ndarray,
        n_corr: int,
        n_frames: int,
        n_elements: int,
    ) -> None:
        """Build the cross-correlation matrix from velocity spectra (tiled)."""
        budget_pairs = self._corr_matrix_budget_pairs(n_frames)
        tiles = self._corr_matrix_tiles(n_elements, budget_pairs)
        fill_tile = partial(
            self._fill_corr_matrix_tile_from_spectra,
            spectra=spectra,
            corr_matrix=corr_matrix,
            win_time=win_time,
            n_corr=n_corr,
            n_frames=n_frames,
        )
        spec = self._phase_parallel("corr_matrix")
        with blas_thread_context(spec.omp_threads):
            max_workers = resolve_n_jobs(spec.n_jobs)
            if max_workers == 1:
                for tile in tiles:
                    fill_tile(tile)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    list(executor.map(fill_tile, tiles))
        self._symmetrize_corr_matrix(corr_matrix, n_elements)

    def _autocorr_diagonal_from_spectra(
        self,
        spectra: np.ndarray,
        win_time: np.ndarray,
        n_corr: int,
        n_frames: int,
    ) -> np.ndarray:
        """Diagonal ``C_ii(omega)`` for DOF rows; ``(n_corr, n_dof)``."""
        cross_freq = np.real(spectra * np.conj(spectra))
        tmp_time = np.real(irfft(cross_freq, n=n_frames, axis=-1))
        windowed = self._window_time_batch(
            tmp_time,
            n_corr,
            n_frames,
            win_time,
            self.lag_symmetrization,
        )
        spectra_bins = np.real(rfft(windowed, axis=-1))
        return spectra_bins.T

    def _build_autocorr_diagonal(
        self,
        spectra: np.ndarray,
        win_time: np.ndarray,
        n_corr: int,
        n_frames: int,
        n_elements: int,
    ) -> np.ndarray:
        """Autocorrelation spectra on the diagonal; shape ``(n_corr, n_elements)``."""
        spec = self._phase_parallel("corr_matrix")

        def build_slice(i0: int, i1: int) -> np.ndarray:
            return self._autocorr_diagonal_from_spectra(
                spectra[i0:i1],
                win_time,
                n_corr,
                n_frames,
            )

        with blas_thread_context(spec.omp_threads):
            max_workers = resolve_n_jobs(spec.n_jobs)
            if max_workers == 1:
                return build_slice(0, n_elements)
            diag = np.empty((n_corr, n_elements), dtype=np.float64)

            def fill_rows(rows: np.ndarray) -> tuple[int, int, np.ndarray]:
                i0, i1 = int(rows[0]), int(rows[-1]) + 1
                return i0, i1, build_slice(i0, i1)

            row_chunks = np.array_split(np.arange(n_elements), max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i0, i1, block in executor.map(fill_rows, row_chunks):
                    diag[:, i0:i1] = block
            return diag

    def _diagonalize_corr_matrix(
        self,
        corr_matrix: np.ndarray,
        freq_indices: np.ndarray,
        n_elements: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_bins = int(freq_indices.size)
        eigenvalues = np.empty((n_bins, n_elements), dtype=np.float64)
        eigenvectors = np.empty(
            (n_bins, n_elements, n_elements), dtype=np.float64
        )
        spec = self._phase_parallel("eigen")
        with blas_thread_context(spec.omp_threads):
            max_workers = resolve_n_jobs(spec.n_jobs)

            def fill_output(out_index: int, freq_index: int) -> None:
                vals, vecs = np.linalg.eigh(corr_matrix[freq_index])
                order = np.argsort(vals)[::-1]
                eigenvalues[out_index] = vals[order]
                eigenvectors[out_index] = vecs[:, order].T

            if max_workers == 1:
                for out_index, freq_index in enumerate(freq_indices):
                    fill_output(out_index, int(freq_index))
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    list(
                        executor.map(
                            lambda pair: fill_output(pair[0], int(pair[1])),
                            enumerate(freq_indices),
                        )
                    )
        return eigenvalues, eigenvectors

    def _init_run_metadata(self) -> None:
        n_elements = self.atomgroup.n_atoms * 3
        self._n_elements = n_elements
        self._n_dof = n_elements - self.n_constraints
        self._sqrt_masses = np.repeat(np.sqrt(self.atomgroup.masses), 3)

    def _init_results_arrays(self) -> None:
        freqs = self._frequency_grid(self.n_corr, self.dt)
        win_norm = 1.0 / np.sqrt(2.0 * np.pi * self.sigma**2)
        win_freq = np.zeros(2 * self.n_corr - 1)
        win_freq[: self.n_corr] = win_norm * np.exp(
            -0.5 * freqs**2 / self.sigma**2
        )
        win_freq[self.n_corr :] = win_freq[self.n_corr - 1 : 0 : -1]
        win_time = np.real(ifft(win_freq))
        n_elements = self._n_elements
        result_fields: dict[str, Any] = {
            "velocities": np.zeros(
                (n_elements, self.n_frames), dtype=np.float64
            ),
            "freqs": freqs,
            "win_time": win_time,
            "n_dof": self._n_dof,
        }
        self.results = Results(**result_fields)

    def _prepare(self):
        """Set things up before the analysis loop begins"""
        self._init_run_metadata()
        self._init_results_arrays()

    def _single_frame(self):
        """Calculate data from a single frame of trajectory"""
        # This runs once for each frame of the trajectory
        # It can contain the main analysis method, or just collect data
        # so that analysis can be done over the aggregate data
        # in _conclude.

        # The trajectory positions update automatically
        velocities = self.atomgroup.velocities
        if velocities is None:
            raise ValueError("Trajectory must contain velocities for FRESEAN.")
        self.results.velocities[:, self._frame_index] = (
            self._sqrt_masses * velocities.flatten()
        )

    def _conclude(self, timings: Optional[dict[str, float]] = None) -> None:
        """Velocity FFT, correlation matrix, normalization, VDOS, and modes."""
        plan = self._run_plan
        if plan is None:
            raise RuntimeError(
                "FRESEAN._conclude requires a run plan from run()."
            )

        if not hasattr(self, "_n_elements"):
            self._init_run_metadata()
        n_elements = self._n_elements
        n_corr = self.n_corr
        win_time = self.results.win_time
        freqs = self.results.freqs
        corr_shape = (n_corr, n_elements, n_elements)
        self._update_freq_indices()

        if plan.calc_velocity_fft is True:
            if not hasattr(self.results, "velocities"):
                raise RuntimeError(
                    "Velocity FFT was planned but time-domain velocities are "
                    "missing; read_traj=True or run a trajectory pass first."
                )
            t_fft = time.perf_counter() if timings is not None else None
            self.results.velocity_spectra = self._rfft_velocities(
                self.results.velocities
            )
            del self.results.velocities
            if t_fft is not None:
                self._add_benchmark_time(
                    timings,
                    BENCH_T_VELOCITY_SPECTRA,
                    time.perf_counter() - t_fft,
                )

        corr_matrix: Optional[np.ndarray] = getattr(
            self.results, "corr_matrix", None
        )
        stored_corr: Optional[np.ndarray] = None
        diag: Optional[np.ndarray] = None

        built_full_corr = False
        built_diag = False
        if plan.calc_full_corr_matrix or plan.calc_autocorr_diagonal is True:
            spectra = getattr(self.results, "velocity_spectra", None)
            if spectra is None or spectra.size == 0:
                raise RuntimeError(
                    "results.velocity_spectra is missing; read_traj=True or "
                    "run a prior step that fills velocity spectra."
                )
            t_corr = time.perf_counter() if timings is not None else None
            if plan.calc_full_corr_matrix:
                if not (
                    isinstance(corr_matrix, np.ndarray)
                    and corr_matrix.shape == corr_shape
                ):
                    corr_matrix = np.empty(corr_shape)
                self._build_corr_matrix(
                    spectra,
                    corr_matrix,
                    win_time,
                    n_corr,
                    self.n_frames,
                    n_elements,
                )
                corr_matrix /= self.n_frames
                built_full_corr = True
            elif plan.calc_autocorr_diagonal is True:
                diag = self._build_autocorr_diagonal(
                    spectra,
                    win_time,
                    n_corr,
                    self.n_frames,
                    n_elements,
                )
                diag /= self.n_frames
                corr_matrix = None
                built_diag = True
            if t_corr is not None and (built_full_corr or built_diag):
                self._add_benchmark_time(
                    timings,
                    BENCH_T_CORR_MATRIX,
                    time.perf_counter() - t_corr,
                )

            if built_full_corr or built_diag:
                t_norm = time.perf_counter() if timings is not None else None
                if built_full_corr:
                    traces = np.trace(corr_matrix, axis1=1, axis2=2)
                else:
                    traces = np.sum(diag, axis=1)
                avg_temp = self._avg_temperature_from_traces(
                    traces, n_corr, win_time, self._n_dof
                )
                vdos_norm = n_corr * win_time[0] * (8.3145 * 0.1 * avg_temp)
                if vdos_norm > 0:
                    if built_full_corr:
                        corr_matrix /= vdos_norm
                    else:
                        diag /= vdos_norm
                self.results.avg_temperature = avg_temp
                self.results.vdos_norm = vdos_norm
                if plan.calc_vdos is True:
                    vidx = self._vdos_freq_idx
                    self.results.vdos_freqs = freqs[vidx]
                    if vdos_norm > 0:
                        self.results.vdos_total = traces[vidx] / vdos_norm
                    else:
                        self.results.vdos_total = traces[vidx].copy()
                if t_norm is not None:
                    self._add_benchmark_time(
                        timings,
                        BENCH_T_VDOS,
                        time.perf_counter() - t_norm,
                    )

        if plan.calc_modes:
            if corr_matrix is None:
                raise RuntimeError(
                    "Mode diagonalization requires a correlation matrix; "
                    "use compute_corr_matrix=True or a prior run that stored "
                    "results.corr_matrix."
                )
            t_eigen = time.perf_counter() if timings is not None else None
            eigenvalues, eigenvectors = self._diagonalize_corr_matrix(
                corr_matrix,
                self._mode_freq_idx,
                n_elements,
            )
            self.results.mode_freqs = freqs[self._mode_freq_idx]
            self.results.eigenvalues = eigenvalues
            self.results.eigenvectors = eigenvectors
            if t_eigen is not None:
                self._add_benchmark_time(
                    timings, BENCH_T_EIGEN, time.perf_counter() - t_eigen
                )
        else:
            self.results.mode_freqs = None
            self.results.eigenvalues = None
            self.results.eigenvectors = None

        if self.compute_corr_matrix and isinstance(corr_matrix, np.ndarray):
            stored_corr = corr_matrix
        self.results.corr_matrix = stored_corr
        self.results.corr_freqs = freqs if stored_corr is not None else None

    def run(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        frames: Optional[Any] = None,
        *,
        read_traj: bool = True,
        compute_vdos: ComputeOption = True,
        compute_corr_matrix: bool = True,
        compute_modes: ComputeOption = True,
        verbose: Optional[bool] = None,
        n_workers: Optional[int] = None,
        n_parts: Optional[int] = None,
        backend: Optional[Union[str, BackendBase]] = None,
        unsupported_backend: bool = False,
        progressbar_kwargs: Optional[dict] = None,
        benchmark: bool = False,
    ):
        """Perform the calculation.

        Parameters
        ----------
        read_traj : bool, optional
            If ``True`` (default), read the trajectory. If ``False``, reuse
            cached results for the same ``fresean_frame_key`` when the run
            plan allows it. Missing intermediates trigger :class:`UserWarning`
            and may still read the trajectory or FFT stored velocities.
        compute_vdos : bool or sequence of float, optional
            ``False`` skips VDOS; ``True`` or a frequency list enables it.
        compute_corr_matrix : bool, optional
            If ``True`` (default), build the full normalized
            ``results.corr_matrix`` and take VDOS from its trace. If
            ``False`` and ``compute_vdos`` is enabled, VDOS is computed from
            diagonal autocorrelation spectra only (``O(n)`` memory in DOF).
            Required when ``compute_modes`` is enabled.
        compute_modes : bool or sequence of float, optional
            ``False`` skips mode diagonalization; ``True`` or a list enables it.
        benchmark : bool, optional
            If ``True``, record wall times for the major FRESEAN phases and
            return them as a dict. The same dict is stored on
            :attr:`FRESEAN.benchmark`. The first ``benchmark=True`` run
            starts from zero per phase; later runs on the same instance
            keep timings for skipped phases (e.g. ``read_traj=False``).
            Phases that run again are reset to zero first, then timed
            (multiple chunks in one phase still sum, e.g. trajectory read
            plus FFT).

            Keys:

            * ``bench_t_read_traj`` — trajectory read (mass-weighted velocities).
            * ``bench_t_velocity_spectra`` — velocity FFT to
              ``results.velocity_spectra``.
            * ``bench_t_corr_matrix`` — correlation build (full matrix or
              diagonal autocorrelation spectra only).
            * ``bench_t_vdos`` — temperature normalization (and ``vdos_total``
              when VDOS is enabled) after a spectral build.
            * ``bench_t_eigen`` — mode diagonalization (:meth:`compute_modes`).
            * ``bench_t_fresean_total`` — sum of the spectral phases above
              (excludes MDAnalysis setup/merge overhead).

        Returns
        -------
        self or dict
            ``self`` when ``benchmark=False``; otherwise a timing dict.
        """
        compute_vdos, self._vdos_freqs_req = self._split_compute_option(
            compute_vdos, "compute_vdos"
        )
        compute_modes, self._mode_freqs_req = self._split_compute_option(
            compute_modes, "compute_modes"
        )
        compute_corr_matrix = bool(compute_corr_matrix)
        self.compute_vdos = compute_vdos
        self.compute_modes = compute_modes
        self.compute_corr_matrix = compute_corr_matrix
        if not compute_modes and not compute_vdos:
            raise ValueError(
                "At least one of compute_modes and compute_vdos must be True"
            )
        if compute_modes and not compute_corr_matrix:
            raise ValueError("compute_modes requires compute_corr_matrix=True")
        self._update_freq_indices()

        frame_key = self._frame_key(start, stop, step, frames)
        n_axes = self.atomgroup.n_atoms * 3
        corr_shape = (self.n_corr, n_axes, n_axes)
        self._run_plan = self._fresean_run_plan(
            read_traj,
            compute_vdos,
            compute_corr_matrix,
            compute_modes,
            frame_key,
            n_axes,
            corr_shape,
        )

        if self._run_plan.calc_read_traj is not True:
            timings = None
            if benchmark:
                timings = self._benchmark_timings_for_run()
                self._reset_fresean_benchmark_phases(timings, self._run_plan)
            self._conclude(timings)
            self.results.fresean_frame_key = frame_key
            if benchmark:
                self.benchmark = finalize_fresean_benchmark(timings)
                return self.benchmark
            return self

        if not benchmark:
            out = super().run(
                start=start,
                stop=stop,
                step=step,
                frames=frames,
                verbose=verbose,
                n_workers=n_workers,
                n_parts=n_parts,
                backend=backend,
                unsupported_backend=unsupported_backend,
                progressbar_kwargs=progressbar_kwargs,
            )
            self.results.fresean_frame_key = frame_key
            return out

        backend = "serial" if backend is None else backend
        progressbar_kwargs = (
            {} if progressbar_kwargs is None else progressbar_kwargs
        )
        if (progressbar_kwargs or verbose) and not (
            backend == "serial" or isinstance(backend, BackendSerial)
        ):
            raise ValueError(
                "Can not display progressbar with non-serial backend"
            )

        if n_workers is None:
            n_workers = (
                backend.n_workers
                if isinstance(backend, BackendBase)
                and hasattr(backend, "n_workers")
                else 1
            )

        n_parts = n_workers if n_parts is None else n_parts

        executor = self._configure_backend(
            backend=backend,
            n_workers=n_workers,
            unsupported_backend=unsupported_backend,
        )
        if hasattr(executor, "n_workers") and n_parts < executor.n_workers:
            warnings.warn(
                (
                    f"Analysis not making use of all workers: "
                    f"{executor.n_workers=} is greater than {n_parts=}"
                )
            )

        worker_func = partial(
            self._compute,
            progressbar_kwargs=progressbar_kwargs,
            verbose=verbose,
        )
        self._setup_frames(
            trajectory=self._trajectory,
            start=start,
            stop=stop,
            step=step,
            frames=frames,
        )
        computation_groups = self._setup_computation_groups(
            start=start, stop=stop, step=step, frames=frames, n_parts=n_parts
        )

        t_read_traj = time.perf_counter()
        remote_objects = executor.apply(worker_func, computation_groups)
        bench_t_read_traj = time.perf_counter() - t_read_traj

        self.frames = np.hstack([obj.frames for obj in remote_objects])
        self.times = np.hstack([obj.times for obj in remote_objects])

        remote_results = [obj.results for obj in remote_objects]
        results_aggregator = self._get_aggregator()
        self.results = results_aggregator.merge(remote_results)

        timings = self._benchmark_timings_for_run()
        self._reset_fresean_benchmark_phases(timings, self._run_plan)
        if self._run_plan.calc_read_traj is True:
            self._add_benchmark_time(
                timings, BENCH_T_READ_TRAJ, bench_t_read_traj
            )
        self._conclude(timings)
        self.results.fresean_frame_key = frame_key
        self.benchmark = finalize_fresean_benchmark(timings)
        return self.benchmark

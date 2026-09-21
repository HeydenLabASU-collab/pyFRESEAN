"""
FRESEAN --- :mod:`pyfresean.analysis.FRESEAN`
===========================================================

This module contains the :class:`FRESEAN` class.

"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np
from MDAnalysis.analysis.backends import BackendBase, BackendSerial

LagSymmetrization = Literal["mirror", "average"]

from MDAnalysis.analysis.base import AnalysisBase, Results
from MDAnalysis.analysis.results import ResultsGroup
from scipy.fft import ifft, irfft, rfft

from pyfresean.benchmark_keys import (
    BENCH_T_CORR_MATRIX,
    BENCH_T_EIGEN,
    BENCH_T_VELOCITY_SPECTRA,
    finalize_fresean_benchmark,
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
    See :meth:`run` for ``read_traj``, ``compute_vdos``, and ``compute_modes``.

    run(..., read_traj=True, compute_vdos=True, compute_modes=True, ...)
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

    # **NOTE**: Add instruction to run parallel
    # export OMP_NUM_THREADS=4
    # taskset -c 0-3 python3 test_fresean.py (=2)
    # python3 test_fresean.py (=2)

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        return ("serial", "multiprocessing", "dask")

    @staticmethod
    def _take_first_result(values):
        return values[0]

    def _needs_corr_matrix(self) -> bool:
        return bool(self.compute_vdos or self.compute_modes)

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
        if self._needs_corr_matrix():
            lookup["corr_matrix"] = FRESEAN._take_first_result
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
        self.compute_modes = True
        self.compute_vdos = True
        self._mode_freqs_req: Optional[Sequence[float]] = None
        self._vdos_freqs_req: Optional[Sequence[float]] = None
        self._update_freq_indices()

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
                            lambda rows: transform_rows(int(rows[0]), int(rows[-1]) + 1),
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
                    tiles.append((i0, i0 + 1, j0, min(j0 + col_size, n_elements)))
                    j0 += col_size
                i0 += 1
        return tiles

    def _symmetrize_corr_matrix(
        self, corr_matrix: np.ndarray, n_elements: int
    ) -> None:
        i_upper, j_upper = np.triu_indices(n_elements, k=1)
        corr_matrix[:, j_upper, i_upper] = corr_matrix[:, i_upper, j_upper]

    def _fill_corr_matrix_tile(
        self,
        tile: tuple[int, int, int, int],
        velocities: np.ndarray,
        corr_matrix: np.ndarray,
        win_time: np.ndarray,
        n_corr: int,
        n_frames: int,
    ) -> None:
        """Fill one upper-triangle tile of ``corr_matrix``."""
        i0, i1, j0, j1 = tile
        row_vel = velocities[i0:i1]
        col_vel = velocities[j0:j1]
        cross_freq = np.real(
            row_vel[:, None, :] * np.conj(col_vel[None, :, :])
        )
        tmp_time = np.real(irfft(cross_freq, n=n_frames, axis=-1))
        windowed = self._window_time_batch(
            tmp_time,
            n_corr,
            n_frames,
            win_time,
            self.lag_symmetrization,
        )
        # windowed is real with length 2*n_corr-1, so rfft yields n_corr bins.
        spectra = np.real(rfft(windowed, axis=-1))
        corr_matrix[:, i0:i1, j0:j1] = np.moveaxis(spectra, -1, 0)

    def _build_corr_matrix(
        self,
        velocities: np.ndarray,
        corr_matrix: np.ndarray,
        win_time: np.ndarray,
        n_corr: int,
        n_frames: int,
        n_elements: int,
    ) -> None:
        """Build the velocity cross-correlation matrix (vectorized tiles)."""
        budget_pairs = self._corr_matrix_budget_pairs(n_frames)
        tiles = self._corr_matrix_tiles(n_elements, budget_pairs)
        fill_tile = partial(
            self._fill_corr_matrix_tile,
            velocities=velocities,
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
        if self._needs_corr_matrix():
            result_fields["corr_matrix"] = np.empty(
                (self.n_corr, n_elements, n_elements)
            )
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
        if not hasattr(self, "_n_elements"):
            self._init_run_metadata()
        n_elements = self._n_elements
        n_corr = self.n_corr
        win_time = self.results.win_time
        freqs = self.results.freqs
        corr_shape = (n_corr, n_elements, n_elements)
        self._update_freq_indices()
        if timings is not None:
            for key in (
                BENCH_T_VELOCITY_SPECTRA,
                BENCH_T_CORR_MATRIX,
                BENCH_T_EIGEN,
            ):
                timings.setdefault(key, 0.0)

        if hasattr(self.results, "velocities"):
            t_fft = time.perf_counter() if timings is not None else None
            self.results.velocity_spectra = self._rfft_velocities(
                self.results.velocities
            )
            del self.results.velocities
            if t_fft is not None:
                timings[BENCH_T_VELOCITY_SPECTRA] += time.perf_counter() - t_fft

        corr_matrix = getattr(self.results, "corr_matrix", None)
        reuse_corr = (
            self.compute_modes
            and not self.compute_vdos
            and isinstance(corr_matrix, np.ndarray)
            and corr_matrix.shape == corr_shape
        )

        if not reuse_corr:
            spectra = getattr(self.results, "velocity_spectra", None)
            if spectra is None or spectra.size == 0:
                raise RuntimeError(
                    "results.velocity_spectra is missing; use read_traj=True."
                )
            t_corr = time.perf_counter() if timings is not None else None
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
            traces = np.trace(corr_matrix, axis1=1, axis2=2)
            avg_temp = self._avg_temperature_from_traces(
                traces, n_corr, win_time, self._n_dof
            )
            vdos_norm = n_corr * win_time[0] * (8.3145 * 0.1 * avg_temp)
            if vdos_norm > 0:
                corr_matrix /= vdos_norm
            self.results.avg_temperature = avg_temp
            self.results.vdos_norm = vdos_norm
            if self.compute_vdos:
                vidx = self._vdos_freq_idx
                self.results.vdos_freqs = freqs[vidx]
                self.results.vdos_total = np.trace(
                    corr_matrix[vidx], axis1=1, axis2=2
                )
            else:
                self.results.vdos_freqs = None
                self.results.vdos_total = None
            if t_corr is not None:
                timings[BENCH_T_CORR_MATRIX] = time.perf_counter() - t_corr

        if self.compute_modes:
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
                timings[BENCH_T_EIGEN] = time.perf_counter() - t_eigen
        else:
            self.results.mode_freqs = None
            self.results.eigenvalues = None
            self.results.eigenvectors = None

        self.results.corr_matrix = corr_matrix
        self.results.corr_freqs = freqs

    def run(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        frames: Optional[Any] = None,
        *,
        read_traj: bool = True,
        compute_vdos: ComputeOption = True,
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
            ``results.corr_matrix`` or ``results.velocity_spectra`` from a
            prior run with the same frame selection when possible.
        compute_vdos : bool or sequence of float, optional
            ``False`` skips VDOS; ``True`` or a frequency list enables it.
        compute_modes : bool or sequence of float, optional
            ``False`` skips mode diagonalization; ``True`` or a list enables it.
        benchmark : bool, optional
            If ``True``, record wall times for the major FRESEAN phases and
            return them as a dict. The same dict is stored on
            :attr:`FRESEAN.benchmark`.

            Keys:

            * ``bench_t_velocity_spectra`` — trajectory read (if any) plus
              velocity FFT to ``results.velocity_spectra``.
            * ``bench_t_corr_matrix`` — correlation matrix, normalization, and
              VDOS (:meth:`compute_vdos`).
            * ``bench_t_eigen`` — mode diagonalization (:meth:`compute_modes`).
            * ``bench_t_spectral`` — sum of the three phases above (excludes
              MDAnalysis setup/merge overhead).

        Returns
        -------
        self or dict
            ``self`` when ``benchmark=False``; otherwise a timing dict.
        """
        self.compute_vdos, self._vdos_freqs_req = self._split_compute_option(
            compute_vdos, "compute_vdos"
        )
        self.compute_modes, self._mode_freqs_req = self._split_compute_option(
            compute_modes, "compute_modes"
        )
        if not self.compute_modes and not self.compute_vdos:
            raise ValueError(
                "At least one of compute_modes and compute_vdos must be True"
            )
        self._update_freq_indices()

        frame_key = self._frame_key(start, stop, step, frames)
        n_axes = self.atomgroup.n_atoms * 3
        corr_shape = (self.n_corr, n_axes, n_axes)
        res = getattr(self, "results", None)
        if (
            not read_traj
            and res is not None
            and getattr(res, "fresean_frame_key", None) == frame_key
        ):
            sp = getattr(res, "velocity_spectra", None)
            cm = getattr(res, "corr_matrix", None)
            has_sp = (
                isinstance(sp, np.ndarray) and sp.size > 0 and sp.shape[0] == n_axes
            )
            has_cm = isinstance(cm, np.ndarray) and cm.shape == corr_shape
            if (self.compute_modes and not self.compute_vdos and has_cm) or (
                self.compute_vdos and has_sp
            ) or (self.compute_modes and has_sp):
                timings = {BENCH_T_VELOCITY_SPECTRA: 0.0} if benchmark else None
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

        t_velocity = time.perf_counter()
        remote_objects = executor.apply(worker_func, computation_groups)
        bench_t_velocity_spectra = time.perf_counter() - t_velocity

        self.frames = np.hstack([obj.frames for obj in remote_objects])
        self.times = np.hstack([obj.times for obj in remote_objects])

        remote_results = [obj.results for obj in remote_objects]
        results_aggregator = self._get_aggregator()
        self.results = results_aggregator.merge(remote_results)

        timings: dict[str, float] = {
            BENCH_T_VELOCITY_SPECTRA: bench_t_velocity_spectra
        }
        self._conclude(timings=timings)
        self.results.fresean_frame_key = frame_key
        self.benchmark = finalize_fresean_benchmark(timings)
        return self.benchmark

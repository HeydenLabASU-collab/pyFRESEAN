"""
FRESEAN --- :mod:`pyfresean.analysis.FRESEAN`
===========================================================

This module contains the :class:`FRESEAN` class.

"""

from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
from MDAnalysis.analysis.backends import BackendBase, BackendSerial

LagSymmetrization = Literal["mirror", "average"]

from MDAnalysis.analysis.base import AnalysisBase, Results
from MDAnalysis.analysis.results import ResultsGroup
from scipy.fft import fft, ifft

from pyfresean.benchmark_keys import (
    BENCH_T_CORR_MATRIX,
    BENCH_T_EIGEN,
    BENCH_T_VELOCITY_MATRIX,
    finalize_fresean_benchmark,
)

if TYPE_CHECKING:
    from MDAnalysis.core.groups import AtomGroup
    from MDAnalysis.core.universe import Universe


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
    n_jobs: int or None
        Number of threads for :meth:`_conclude` work. ``1`` or ``None`` runs
        serially (default). ``-1`` uses :func:`os.cpu_count`. Used for the
        velocity cross-correlation loop (parallel over row index ``i``) and
        for per-frequency :func:`numpy.linalg.eigh` diagonalization.
    run(..., n_workers=N, backend="multiprocessing")
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
        results of calculation are stored here, after calling
        :meth:`FRESEAN.run`
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

    def _get_aggregator(self):
        return ResultsGroup(
            lookup={
                "velocities": ResultsGroup.ndarray_hstack,
                "freqs": FRESEAN._take_first_result,
                "win_time": FRESEAN._take_first_result,
                "n_dof": FRESEAN._take_first_result,
                "corr_matrix": FRESEAN._take_first_result,
            }
        )

    def __init__(
        self,
        universe_or_atomgroup: Union["Universe", "AtomGroup"],
        select: str = "all",
        n_constraints: int = 0,
        n_corr: int = 500,
        dt: float = 0.004,
        sigma: float = 10.0,
        lag_symmetrization: LagSymmetrization = "mirror",
        n_jobs: Optional[int] = 1,
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
        if n_jobs is not None and n_jobs != 1 and n_jobs != -1 and n_jobs < 2:
            raise ValueError(
                "n_jobs must be None, 1, -1, or an integer >= 2; "
                f"got {n_jobs!r}"
            )
        self.n_jobs = n_jobs
        self.benchmark: Optional[dict[str, float]] = None

    @staticmethod
    def _resolve_n_jobs(n_jobs: Optional[int]) -> int:
        if n_jobs is None or n_jobs == 1:
            return 1
        if n_jobs < 0:
            return os.cpu_count() or 1
        return n_jobs

    def _build_windowed_lags(
        self,
        tmp_time: np.ndarray,
        n_corr: int,
        n_frames: int,
    ) -> np.ndarray:
        """Pack symmetrized lags into the length ``2 * n_corr - 1`` FFT input."""
        tmp_windowed = np.zeros(2 * n_corr - 1, dtype=np.float64)
        if self.lag_symmetrization == "average":
            tmp_windowed[0] = tmp_time[0]
            for k in range(1, n_corr):
                tmp_windowed[k] = (tmp_time[k] + tmp_time[n_frames - k]) / 2.0
        else:
            tmp_windowed[:n_corr] = tmp_time[:n_corr]
        tmp_windowed[n_corr:] = tmp_windowed[n_corr - 1 : 0 : -1]
        return tmp_windowed

    def _corr_spectrum_from_velocities(
        self,
        vel_i: np.ndarray,
        vel_j: np.ndarray,
        win_time: np.ndarray,
        n_corr: int,
        n_frames: int,
    ) -> np.ndarray:
        tmp_freq = np.real(vel_i * vel_j.conj())
        tmp_time = np.real(ifft(tmp_freq))
        tmp_windowed = self._build_windowed_lags(tmp_time, n_corr, n_frames)
        tmp_windowed *= win_time
        return np.real(fft(tmp_windowed)[:n_corr])

    def _fill_corr_matrix_row(
        self,
        i: int,
        velocities: np.ndarray,
        corr_matrix: np.ndarray,
        win_time: np.ndarray,
        n_corr: int,
        n_frames: int,
        n_elements: int,
    ) -> None:
        for j in range(i, n_elements):
            corr_matrix[:, i, j] = self._corr_spectrum_from_velocities(
                velocities[i],
                velocities[j],
                win_time,
                n_corr,
                n_frames,
            )

    def _symmetrize_corr_matrix(
        self, corr_matrix: np.ndarray, n_elements: int
    ) -> None:
        for i in range(n_elements):
            for j in range(i + 1, n_elements):
                corr_matrix[:, j, i] = corr_matrix[:, i, j]

    def _build_corr_matrix(
        self,
        velocities: np.ndarray,
        corr_matrix: np.ndarray,
        win_time: np.ndarray,
        n_corr: int,
        n_frames: int,
        n_elements: int,
    ) -> None:
        max_workers = self._resolve_n_jobs(self.n_jobs)
        if max_workers == 1:
            for i in range(n_elements):
                self._fill_corr_matrix_row(
                    i,
                    velocities,
                    corr_matrix,
                    win_time,
                    n_corr,
                    n_frames,
                    n_elements,
                )
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(
                    executor.map(
                        lambda i: self._fill_corr_matrix_row(
                            i,
                            velocities,
                            corr_matrix,
                            win_time,
                            n_corr,
                            n_frames,
                            n_elements,
                        ),
                        range(n_elements),
                    )
                )
        self._symmetrize_corr_matrix(corr_matrix, n_elements)

    def _fill_eigen_at_frequency(
        self,
        freq_index: int,
        corr_matrix: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
    ) -> None:
        vals, vecs = np.linalg.eigh(corr_matrix[freq_index])
        order = np.argsort(vals)[::-1]
        eigenvalues[freq_index] = vals[order]
        eigenvectors[freq_index] = vecs[:, order].T

    def _diagonalize_corr_matrix(
        self,
        corr_matrix: np.ndarray,
        n_corr: int,
        n_elements: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        eigenvalues = np.empty((n_corr, n_elements), dtype=np.float64)
        eigenvectors = np.empty(
            (n_corr, n_elements, n_elements), dtype=np.float64
        )
        max_workers = self._resolve_n_jobs(self.n_jobs)
        if max_workers == 1:
            for freq_index in range(n_corr):
                self._fill_eigen_at_frequency(
                    freq_index,
                    corr_matrix,
                    eigenvalues,
                    eigenvectors,
                )
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(
                    executor.map(
                        lambda freq_index: self._fill_eigen_at_frequency(
                            freq_index,
                            corr_matrix,
                            eigenvalues,
                            eigenvectors,
                        ),
                        range(n_corr),
                    )
                )
        return eigenvalues, eigenvectors

    def _init_run_metadata(self) -> None:
        n_elements = self.atomgroup.n_atoms * 3
        self._n_elements = n_elements
        self._n_dof = n_elements - self.n_constraints
        self._sqrt_masses = np.repeat(np.sqrt(self.atomgroup.masses), 3)

    def _init_results_arrays(self) -> None:
        wn0 = 1.0 / ((2 * self.n_corr - 1) * self.dt) * 33.3564
        freqs = np.arange(self.n_corr) * wn0
        win_norm = 1.0 / np.sqrt(2.0 * np.pi * self.sigma**2)
        win_freq = np.zeros(2 * self.n_corr - 1)
        win_freq[: self.n_corr] = win_norm * np.exp(
            -0.5 * freqs**2 / self.sigma**2
        )
        win_freq[self.n_corr :] = win_freq[self.n_corr - 1 : 0 : -1]
        win_time = np.real(ifft(win_freq))
        n_elements = self._n_elements
        self.results = Results(
            velocities=np.zeros(
                (n_elements, self.n_frames), dtype=np.complex128
            ),
            corr_matrix=np.empty((self.n_corr, n_elements, n_elements)),
            freqs=freqs,
            win_time=win_time,
            n_dof=self._n_dof,
        )

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
        """Calculate the final results of the analysis."""
        if not hasattr(self, "_n_elements"):
            self._init_run_metadata()
        n_elements = self._n_elements
        n_frames = self.n_frames
        n_corr = self.n_corr

        if timings is not None:
            t_corr = time.perf_counter()

        velocities = fft(self.results.velocities, axis=1)
        corr_matrix = self.results.corr_matrix
        win_time = self.results.win_time

        self._build_corr_matrix(
            velocities,
            corr_matrix,
            win_time,
            n_corr,
            n_frames,
            n_elements,
        )
        corr_matrix /= n_frames

        if timings is not None:
            timings[BENCH_T_CORR_MATRIX] = time.perf_counter() - t_corr
            t_eigen = time.perf_counter()

        eigenvalues, eigenvectors = self._diagonalize_corr_matrix(
            corr_matrix,
            n_corr,
            n_elements,
        )

        avg_temp = (
            (np.sum(eigenvalues[0]) + 2 * np.sum(eigenvalues[1:]))
            / (2 * n_corr - 1)
            / win_time[0]
            / (8.3145 * 0.1)
            / self._n_dof
        )
        vdos_norm = n_corr * win_time[0] * (8.3145 * 0.1 * avg_temp)
        if vdos_norm > 0:
            eigenvalues /= vdos_norm
            corr_matrix /= vdos_norm

        self.results.eigenvalues = eigenvalues
        self.results.eigenvectors = eigenvectors
        self.results.avg_temperature = avg_temp
        self.results.vdos_norm = vdos_norm
        self.results.vdos_total = np.sum(eigenvalues, axis=1)
        del self.results.velocities

        if timings is not None:
            timings[BENCH_T_EIGEN] = time.perf_counter() - t_eigen

    def run(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        frames: Optional[Any] = None,
        verbose: Optional[bool] = None,
        n_workers: Optional[int] = None,
        n_parts: Optional[int] = None,
        backend: Optional[Union[str, BackendBase]] = None,
        *,
        unsupported_backend: bool = False,
        progressbar_kwargs: Optional[dict] = None,
        benchmark: bool = False,
    ):
        """Perform the calculation.

        Parameters
        ----------
        benchmark : bool, optional
            If ``True``, record wall times for the major FRESEAN phases and
            return them as a dict. The same dict is stored on
            :attr:`FRESEAN.benchmark`.

            Keys:

            * ``bench_t_velocity_matrix`` — trajectory loop collecting
              mass-weighted velocities (:meth:`_single_frame`).
            * ``bench_t_corr_matrix`` — velocity FFT and correlation-matrix
              assembly in :meth:`_conclude`.
            * ``bench_t_eigen`` — per-frequency diagonalization and VDOS
              normalization in :meth:`_conclude`.
            * ``bench_t_spectral`` — sum of the three phases above (excludes
              MDAnalysis setup/merge overhead).

        Returns
        -------
        self or dict
            ``self`` when ``benchmark=False``; otherwise a timing dict.
        """
        if not benchmark:
            return super().run(
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
        bench_t_velocity_matrix = time.perf_counter() - t_velocity

        self.frames = np.hstack([obj.frames for obj in remote_objects])
        self.times = np.hstack([obj.times for obj in remote_objects])

        remote_results = [obj.results for obj in remote_objects]
        results_aggregator = self._get_aggregator()
        self.results = results_aggregator.merge(remote_results)

        timings: dict[str, float] = {
            BENCH_T_VELOCITY_MATRIX: bench_t_velocity_matrix
        }
        self._conclude(timings=timings)
        self.benchmark = finalize_fresean_benchmark(timings)
        return self.benchmark

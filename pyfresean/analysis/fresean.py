"""
FRESEAN --- :mod:`pyfresean.analysis.FRESEAN`
===========================================================

This module contains the :class:`FRESEAN` class.

"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

LagSymmetrization = Literal["mirror", "average"]

import numpy as np
from MDAnalysis.analysis.base import AnalysisBase, Results
from scipy.fft import fft, ifft

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

    def __init__(
        self,
        universe_or_atomgroup: Union["Universe", "AtomGroup"],
        select: str = "all",
        n_constraints: int = 0,
        n_corr: int = 500,
        dt: float = 0.004,
        sigma: float = 10.0,
        lag_symmetrization: LagSymmetrization = "mirror",
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

    def _prepare(self):
        """Set things up before the analysis loop begins"""
        # This is an optional method that runs before
        # _single_frame loops over the trajectory.
        # It is useful for setting up results arrays
        n_elements = self.atomgroup.n_atoms * 3
        self._n_elements = n_elements
        self._n_dof = n_elements - self.n_constraints
        self._sqrt_masses = np.repeat(np.sqrt(self.atomgroup.masses), 3)

        wn0 = 1.0 / ((2 * self.n_corr - 1) * self.dt) * 33.3564
        freqs = np.arange(self.n_corr) * wn0
        win_norm = 1.0 / np.sqrt(2.0 * np.pi * self.sigma**2)
        win_freq = np.zeros(2 * self.n_corr - 1)
        win_freq[: self.n_corr] = win_norm * np.exp(
            -0.5 * freqs**2 / self.sigma**2
        )
        win_freq[self.n_corr :] = win_freq[self.n_corr - 1 : 0 : -1]
        win_time = np.real(ifft(win_freq))

        self.results = Results(
            velocities=np.zeros((n_elements, self.n_frames), dtype=np.complex128),
            corr_matrix=np.empty((self.n_corr, n_elements, n_elements)),
            freqs=freqs,
            win_time=win_time,
            n_dof=self._n_dof,
        )

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

    def _conclude(self):
        """Calculate the final results of the analysis"""
        # This is an optional method that runs after
        # _single_frame loops over the trajectory.
        # It is useful for calculating the final results
        # of the analysis.
        n_elements = self._n_elements
        n_frames = self.n_frames
        n_corr = self.n_corr

        velocities = fft(self.results.velocities, axis=1)
        corr_matrix = self.results.corr_matrix
        win_time = self.results.win_time

        for i in range(n_elements):
            for j in range(i, n_elements):
                tmp_freq = np.real(velocities[i] * velocities[j].conj())
                tmp_time = np.real(ifft(tmp_freq))
                tmp_windowed = self._build_windowed_lags(tmp_time, n_corr, n_frames)
                tmp_windowed *= win_time
                corr_matrix[:, i, j] = np.real(fft(tmp_windowed)[:n_corr])
                if i != j:
                    corr_matrix[:, j, i] = corr_matrix[:, i, j]

        corr_matrix /= n_frames

        eigenvalues = np.empty((n_corr, n_elements), dtype=np.float64)
        eigenvectors = np.empty((n_corr, n_elements, n_elements), dtype=np.float64)
        for freq_index in range(n_corr):
            vals, vecs = np.linalg.eigh(corr_matrix[freq_index])
            order = np.argsort(vals)[::-1]
            eigenvalues[freq_index] = vals[order]
            eigenvectors[freq_index] = vecs[:, order].T

        avg_temp = (
            np.sum(eigenvalues[0])
            + 2 * np.sum(eigenvalues[1:])
        ) / (2 * n_corr - 1) / win_time[0] / (8.3145 * 0.1) / self._n_dof
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

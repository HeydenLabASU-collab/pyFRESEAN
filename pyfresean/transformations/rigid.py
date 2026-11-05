"""Remove rigid translation and rotation from atom velocities."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.transformations.base import TransformationBase

from pyfresean.exceptions import MissingVelocityWarning

if TYPE_CHECKING:
    from MDAnalysis.core.groups import AtomGroup

# Eigenvalues below this fraction of the largest (eigenvalue) moment are a null axis and ignored.
_INERTIA_CUTOFF = 1e-12


def _mass_weighted_mean(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mass-weighted mean of positions or of velocities."""
    return np.dot(weights, values) / float(weights.sum())


def _moment_of_inertia(relative: np.ndarray, weights: np.ndarray) -> np.ndarray:
    r2 = np.einsum("ij,ij->i", relative, relative)
    tensor = np.eye(3) * np.dot(weights, r2)
    tensor -= np.einsum("i,ij,ik->jk", weights, relative, relative)
    return tensor


def _angular_velocity(inertia: np.ndarray, angular: np.ndarray) -> np.ndarray:
    """Angular velocity ``ω = I^{-1} L`` about the center of mass.

    A zero principal moment means the atoms have no extent along that
    axis, so that component of ``ω`` is left at zero.
    """
    moments, axes = np.linalg.eigh(inertia)
    cutoff = _INERTIA_CUTOFF * float(moments[-1])
    omega = np.zeros(3, dtype=np.float64)
    for moment, axis in zip(moments, axes.T):
        if moment > cutoff:
            omega += axis * (np.dot(axis, angular) / moment)
    return omega


def _rigid_body_velocity(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """Translational plus rotational velocity of one rigid body.

    The result is ``v_COM + ω × (r - r_COM)`` for each atom.
    """
    weights = np.asarray(masses, dtype=np.float64).reshape(-1)
    if (
        weights.size == 0
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("masses must be positive and finite")
    coords = np.asarray(positions, dtype=np.float64)
    speeds = np.asarray(velocities, dtype=np.float64)
    v_com = _mass_weighted_mean(weights, speeds)
    relative = coords - _mass_weighted_mean(weights, coords)
    angular = np.dot(weights, np.cross(relative, speeds - v_com))
    omega = _angular_velocity(_moment_of_inertia(relative, weights), angular)
    return v_com + np.cross(omega, relative)


class RemoveRigidVelocity(TransformationBase):
    """Remove center-of-mass translation and rotation from velocities.

    Each frame subtracts ``v_COM`` and ``ω × (r - r_COM)`` from the
    selected atoms. The inertia tensor is taken about the center of
    mass, so all three rotational directions are removed. Coordinates
    are left unchanged. The selection must already be whole; a
    periodic jump makes the center of mass and the inertia tensor
    meaningless.
    """

    def __init__(
        self,
        atomgroup: AtomGroup,
        max_threads: int = 1,
        parallelizable: bool = True,
    ):
        super().__init__(max_threads=max_threads, parallelizable=parallelizable)
        self.atomgroup = atomgroup
        self.universe = atomgroup.universe
        self._missing_velocity_warned = False

    def __call__(self, ts):
        if not ts.has_velocities:
            if not self._missing_velocity_warned:
                warnings.warn(
                    "RemoveRigidVelocity: frame has no velocities; "
                    "skipping rigid-velocity removal",
                    MissingVelocityWarning,
                    stacklevel=2,
                )
                self._missing_velocity_warned = True
            return ts
        selection = self.atomgroup
        try:
            masses = np.asarray(selection.masses, dtype=np.float64)
        except NoDataError as exc:
            raise ValueError(
                "RemoveRigidVelocity requires atom masses"
            ) from exc
        correction = _rigid_body_velocity(
            selection.positions, selection.velocities, masses
        )
        selection.velocities = selection.velocities - correction
        return ts

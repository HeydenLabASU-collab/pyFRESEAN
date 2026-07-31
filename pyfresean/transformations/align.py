from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.transformations.base import TransformationBase

from pyfresean.exceptions import MissingBoxWarning

if TYPE_CHECKING:
    from MDAnalysis.core.groups import AtomGroup


class Align(TransformationBase):
    def __init__(
        self,
        atomgroup: AtomGroup,
        reference_positions: Optional[np.ndarray] = None,
        align: bool = True,
        center_com: bool = True,
        subtract_com_velocity: bool = False,
        place_com_in_box: bool = True,
        rotate: bool = True,
        rotate_velocities: bool = True,
        max_threads: int = 1,
        parallelizable: bool = True,
    ):
        super().__init__(max_threads=max_threads, parallelizable=parallelizable)
        self.atomgroup = atomgroup
        self.universe = atomgroup.universe
        self.reference_positions = reference_positions
        self.align = align
        self.center_com = center_com
        self.subtract_com_velocity = subtract_com_velocity
        self.place_com_in_box = place_com_in_box
        self.rotate = rotate
        self.rotate_velocities = rotate_velocities
        self._missing_box_warned = False

    def _ensure_box(self) -> None:
        """Install a dummy box only when wrapping needs periodic dimensions."""
        if self.universe.dimensions is None:
            if not self._missing_box_warned:
                warnings.warn(
                    "Align: no box dimensions in trajectory; "
                    "using cubic (100 Å)³ dummy box for wrapping",
                    MissingBoxWarning,
                    stacklevel=2,
                )
                self._missing_box_warned = True
            self.universe.dimensions = np.array(
                [100.0, 100.0, 100.0, 90.0, 90.0, 90.0],
                dtype=np.float32,
            )

    def _align_frame(self) -> None:
        u = self.universe
        sel = self.atomgroup
        # Kabsch alignment and COM centering do not need box dimensions;
        # only MDAnalysis wrap() (place_com_in_box=True) does.
        if self.place_com_in_box:
            self._ensure_box()

        ref_pos = (
            sel.positions
            if self.reference_positions is None
            else self.reference_positions
        )
        weights = sel.masses
        sel_com = sel.center_of_mass()
        ref_com = np.average(ref_pos, axis=0, weights=weights)

        if self.align and self.rotate:
            rot, _ = rotation_matrix(
                sel.positions - sel_com,
                ref_pos - ref_com,
                weights=weights,
            )
        else:
            rot = np.identity(3)

        if self.center_com:
            u.atoms.translate(-sel_com)
            if self.rotate:
                u.atoms.rotate(rot)
            u.atoms.translate(ref_com)
        elif self.rotate:
            u.atoms.positions = np.dot(u.atoms.positions - sel_com, rot.T) + sel_com

        ts = u.trajectory.ts
        if self.rotate_velocities and self.rotate and ts.has_velocities:
            u.atoms.velocities = np.dot(u.atoms.velocities, rot.T)

        if self.subtract_com_velocity and ts.has_velocities:
            com_vel = np.average(sel.velocities, axis=0, weights=weights)
            u.atoms.velocities -= com_vel

        if self.place_com_in_box and u.dimensions is not None:
            u.atoms.wrap(compound="fragments")

    def __call__(self, ts):
        self._align_frame()
        return ts

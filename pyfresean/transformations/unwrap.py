from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.mdamath import make_whole
from MDAnalysis.transformations.base import TransformationBase

from pyfresean.exceptions import MissingBoxWarning

if TYPE_CHECKING:
    from MDAnalysis.core.groups import AtomGroup


class Unwrap(TransformationBase):
    def __init__(
        self,
        atomgroup: AtomGroup,
        max_threads: int = 1,
        parallelizable: bool = True,
    ):
        super().__init__(max_threads=max_threads, parallelizable=parallelizable)
        self.atomgroup = atomgroup
        self.universe = atomgroup.universe
        self._missing_box_warned = False

    def __call__(self, ts):
        if self.universe.dimensions is None:
            if not self._missing_box_warned:
                warnings.warn(
                    "Unwrap: no box dimensions available; skipping unwrap",
                    MissingBoxWarning,
                    stacklevel=2,
                )
                self._missing_box_warned = True
            return ts

        try:
            bonds = self.universe.bonds
        except NoDataError:
            return ts

        if len(bonds) == 0:
            return ts

        for fragment in self.atomgroup.fragments:
            make_whole(fragment)
        return ts

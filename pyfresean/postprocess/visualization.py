"""Draw Cartesian mode displacements and a one-period vibration."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Optional

import numpy as np
from MDAnalysis.coordinates.PDB import PDBWriter

if TYPE_CHECKING:
    from MDAnalysis.core.groups import AtomGroup

_STYLES = {
    "cartoon": {"cartoon": {"color": "spectrum"}},
    "stick": {"stick": {"radius": 0.15}},
    "cartoon+stick": {
        "cartoon": {"color": "spectrum"},
        "stick": {"radius": 0.12},
    },
}
# Shaft radius in angstroms. Arrow length follows the displacement.
_ARROW_RADIUS = 0.24


def _as_vectors(array: np.ndarray, name: str) -> np.ndarray:
    vectors = np.asarray(array, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"expected {name} shape (n, 3), got {vectors.shape}")
    return vectors


def _display_scale(
    displacements: np.ndarray, max_length: Optional[float]
) -> float:
    if max_length is None:
        return 1.0
    if max_length <= 0.0 or not np.isfinite(max_length):
        raise ValueError("max_length must be positive and finite")
    longest = float(np.linalg.norm(displacements, axis=1).max())
    if longest == 0.0 or not np.isfinite(longest):
        return 1.0
    return float(max_length) / longest


def scale_to_max_length(
    displacements: np.ndarray, max_length: float = 3.0
) -> np.ndarray:
    """Rescale so the longest vector has length ``max_length``."""
    vectors = _as_vectors(displacements, "displacements")
    return vectors * _display_scale(vectors, max_length)


def oscillation_frames(
    positions: np.ndarray,
    displacements: np.ndarray,
    n_frames: int = 32,
) -> np.ndarray:
    """One period of motion along each displacement.

    Frame ``k`` is the reference coordinate plus
    ``sin(2π k / n_frames)`` times that atom's displacement. The
    structure leaves the reference along ``+d``, returns, leaves along
    ``-d``, and returns. Each atom follows its own vector.
    """
    coords = _as_vectors(positions, "positions")
    disp = _as_vectors(displacements, "displacements")
    if coords.shape != disp.shape:
        raise ValueError(
            f"positions shape {coords.shape} != "
            f"displacements shape {disp.shape}"
        )
    if n_frames < 2:
        raise ValueError("n_frames must be at least 2")
    phase = 2.0 * np.pi * np.arange(n_frames) / n_frames
    sine = np.sin(phase)[:, None, None]
    return coords[None, :, :] + disp[None, :, :] * sine


def _pdb_models(atomgroup: AtomGroup, frames: np.ndarray) -> str:
    saved = np.array(atomgroup.positions, copy=True)
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as handle:
            path = handle.name
        with PDBWriter(
            path,
            multiframe=True,
            n_atoms=atomgroup.n_atoms,
            bonds="all",
        ) as writer:
            for coords in frames:
                atomgroup.positions = coords
                writer.write(atomgroup)
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    finally:
        atomgroup.positions = saved
        if path is not None:
            os.unlink(path)


def _arrow_geometry(
    atomgroup: AtomGroup,
    displacements: np.ndarray,
    arrow_positions: Optional[np.ndarray],
    arrow_displacements: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if arrow_positions is None and arrow_displacements is None:
        starts = np.array(atomgroup.positions, copy=True, dtype=np.float64)
        return starts, displacements
    if arrow_positions is None or arrow_displacements is None:
        raise ValueError(
            "arrow_positions and arrow_displacements must be given together"
        )
    starts = _as_vectors(arrow_positions, "arrow_positions")
    vectors = _as_vectors(arrow_displacements, "arrow_displacements")
    if starts.shape != vectors.shape:
        raise ValueError(
            "arrow_positions and arrow_displacements must have the same shape"
        )
    return starts, vectors


def _arrow_spec(
    start: np.ndarray, end: np.ndarray, color: str, radius: float
) -> dict:
    return {
        "start": {
            "x": float(start[0]),
            "y": float(start[1]),
            "z": float(start[2]),
        },
        "end": {"x": float(end[0]), "y": float(end[1]), "z": float(end[2])},
        "radius": radius,
        "color": color,
    }


def _add_arrows(view, starts: np.ndarray, vectors: np.ndarray, radius: float):
    for start, vector in zip(starts, vectors):
        if float(np.linalg.norm(vector)) < 1e-8:
            continue
        view.addArrow(_arrow_spec(start, start + vector, "orange", radius))
        view.addArrow(_arrow_spec(start, start - vector, "gold", radius))


def _render(  # pylint: disable=too-many-arguments
    pdb: str,
    *,
    animate: bool,
    arrows: bool,
    starts: np.ndarray,
    vectors: np.ndarray,
    style: str,
    radius: float,
    width: int,
    height: int,
):
    try:
        import py3Dmol  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "visualize_mode_displacement requires py3Dmol. "
            'Install it with: pip install "pyfresean[visualization]"'
        ) from exc

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModelsAsFrames(pdb, "pdb")
    if animate:
        viewer.animate({"loop": "forward", "reps": 0, "interval": 80})
    try:
        spec = _STYLES[style]
    except KeyError as exc:
        names = ", ".join(repr(name) for name in _STYLES)
        raise ValueError(f"style must be one of {names}") from exc
    viewer.setStyle(spec)
    viewer.setBackgroundColor("white")
    if arrows:
        _add_arrows(viewer, starts, vectors, radius)
    viewer.zoomTo()
    return viewer


def visualize_mode_displacement(  # pylint: disable=too-many-arguments,too-many-locals
    atomgroup: AtomGroup,
    displacements: np.ndarray,
    *,
    n_frames: int = 32,
    animate: bool = True,
    arrows: bool = True,
    max_length: Optional[float] = 3.0,
    style: str = "cartoon",
    arrow_positions: Optional[np.ndarray] = None,
    arrow_displacements: Optional[np.ndarray] = None,
    width: int = 600,
    height: int = 450,
):
    """Visualize a mode displacement on ``atomgroup`` with py3Dmol.

    ``displacements`` are Cartesian, as from
    :func:`pyfresean.postprocess.cartesian_displacements`. ``max_length``
    rescales them so the longest vector has that length in ångströms, and
    the same factor is applied to ``arrow_displacements``. Arrows are drawn
    along ``+d`` and ``-d``. Their length is that displacement. Every
    arrow uses the same shaft radius. With ``animate=True`` each atom travels
    from the reference position out along its arrow and back, then out
    along the opposite arrow and back. The player repeats that period
    forward.

    Pass ``arrow_positions`` and ``arrow_displacements`` together to draw
    arrows somewhere other than on every atom, for example one arrow at
    each coarse-grained bead center. ``style`` is ``cartoon``, ``stick``,
    or ``cartoon+stick``. ``cartoon+stick`` draws the ribbon and a thin
    stick for each bond. The returned viewer displays in a notebook
    through ``IPython.display.display``.
    """
    vectors = _as_vectors(displacements, "displacements")
    if vectors.shape[0] != atomgroup.n_atoms:
        raise ValueError(
            f"expected displacements for {atomgroup.n_atoms} atoms, "
            f"got {vectors.shape[0]}"
        )
    scale = _display_scale(vectors, max_length)
    vectors = vectors * scale
    starts, arrow_vectors = _arrow_geometry(
        atomgroup, vectors, arrow_positions, arrow_displacements
    )
    if arrow_displacements is not None:
        arrow_vectors = arrow_vectors * scale
    frames = (
        oscillation_frames(np.array(atomgroup.positions), vectors, n_frames)
        if animate
        else np.array(atomgroup.positions, dtype=np.float64)[None, :, :]
    )
    pdb = _pdb_models(atomgroup, frames)
    return _render(
        pdb,
        animate=animate,
        arrows=arrows,
        starts=starts,
        vectors=arrow_vectors,
        style=style,
        radius=_ARROW_RADIUS,
        width=width,
        height=height,
    )

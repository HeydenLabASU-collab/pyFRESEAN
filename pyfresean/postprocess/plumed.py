"""PLUMED DIRECTION vectors, the mode PDB, and the projection script."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np

from pyfresean.exceptions import PlumedAllAtomWarning
from pyfresean.postprocess.spectral import mode_components

_AA_MODE_MESSAGE = (
    "PLUMED directions are for all-atom coordinates. This mode length "
    "does not match the atom count; backmap coarse-grained modes with "
    "backmap_modes first."
)


def _particle_count(mode_vectors: np.ndarray) -> Optional[int]:
    vectors = np.asarray(mode_vectors)
    if vectors.ndim == 2 and vectors.shape[-1] == 3:
        return int(vectors.shape[0])
    if vectors.ndim == 1 and vectors.size % 3 == 0 and vectors.size:
        return int(vectors.size // 3)
    return None


def plumed_directions(
    mode_vectors: np.ndarray,
    masses: np.ndarray,
    scale: float = 100.0,
    n_atoms: Optional[int] = None,
) -> np.ndarray:
    """Mass-weighted mode to PLUMED ``DIRECTION`` components.

    ``A_i = scale * u_i * sqrt(m_i)``. PLUMED uses these as forces, so
    the acceleration follows the Cartesian direction. ``scale`` defaults
    to 100 for PDB ``%8.3f``. ``masses`` are the masses of the particles
    in the mode.

    Pass ``n_atoms`` when those particles should be the atoms in a
    structure. A different length warns and still returns the vector.
    Backmap a coarse-grained mode before using it on the all-atom system.
    """
    n_particles = _particle_count(mode_vectors)
    if n_atoms is not None and n_particles not in (None, n_atoms):
        warnings.warn(_AA_MODE_MESSAGE, PlumedAllAtomWarning, stacklevel=2)
    vectors, masses = mode_components(mode_vectors, masses)
    return scale * vectors * np.sqrt(masses)[:, np.newaxis]


def _format_pdb_atom_line(atom, coords: Sequence[float]) -> str:
    record = str(getattr(atom, "record_type", "") or "ATOM")
    chain = str(getattr(atom, "chainID", "") or getattr(atom, "segid", "") or " ")
    x, y, z = coords
    return (
        f"{record:<6}{int(atom.index) + 1:5d} "
        f"{atom.name:>4s} {atom.resname:>3s}{chain[:1]}"
        f"{int(atom.resid):4d}    "
        f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
        f"  1.00  0.00\n"
    )


def _write_model(writer, atomgroup, remark: str, coordinates=None) -> None:
    """One PDB model. ``coordinates`` defaults to the current positions."""
    writer.write(f"REMARK TYPE={remark}\n")
    if coordinates is None:
        pairs = ((atom, atom.position) for atom in atomgroup)
    else:
        pairs = zip(atomgroup, coordinates)
    for atom, coords in pairs:
        writer.write(_format_pdb_atom_line(atom, coords))
    writer.write("END\n")


def write_plumed_mode_input(
    atomgroup,
    aa_modes: Sequence[np.ndarray],
    path: str,
    *,
    scale: float = 100.0,
    include_reference: bool = True,
) -> None:
    """Write one PLUMED PDB: optional reference, then one direction per mode.

    ``include_reference=True`` starts with ``REMARK TYPE=OPTIMAL`` and the
    current coordinates. Each mode is a ``REMARK TYPE=DIRECTION`` model.
    One mode and ``include_reference=False`` is a single direction file.
    Masses and the atom count come from ``atomgroup``.
    """
    if len(aa_modes) == 0:
        raise ValueError("aa_modes must contain at least one mode")
    masses = np.asarray(atomgroup.masses, dtype=np.float64)
    directions = [
        plumed_directions(
            aa_vectors,
            masses,
            scale=scale,
            n_atoms=atomgroup.n_atoms,
        )
        for aa_vectors in aa_modes
    ]

    with open(path, "w", encoding="utf-8") as writer:
        if include_reference:
            _write_model(writer, atomgroup, "OPTIMAL")
        for mode_directions in directions:
            _write_model(writer, atomgroup, "DIRECTION", mode_directions)


def write_plumed_projection_input(
    path: str,
    n_modes: int,
    *,
    reference: str = "plumed-mode-input.pdb",
    output: str = "plumed-mode-projection.out",
    stride: int = 1000,
) -> None:
    """Write the PLUMED script that loads the mode PDB and prints projections.

    ``plumed driver --plumed`` and ``gmx mdrun -plumed`` read this file.
    The PDB named by ``reference`` holds the structure and directions.
    Printed components are ``pca.eig-1`` through ``pca.eig-<n_modes>``.
    """
    if n_modes < 1:
        raise ValueError("n_modes must be at least 1")
    components = ",".join(f"pca.eig-{index}" for index in range(1, n_modes + 1))
    text = (
        "# Import the file containing the eigenvectors\n"
        f"PCAVARS REFERENCE={reference} TYPE=OPTIMAL LABEL=pca\n"
        "\n"
        "# Print the displacement projections\n"
        f"PRINT ARG={components} FILE={output} STRIDE={stride}\n"
    )
    with open(path, "w", encoding="utf-8") as writer:
        writer.write(text)

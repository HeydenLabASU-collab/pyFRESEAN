"""PLUMED DIRECTION vectors, the mode PDB, and PLUMED scripts."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np

from pyfresean.exceptions import PlumedAllAtomWarning
from pyfresean.postprocess.spectral import mode_components

# PLUMED reads PDB coordinates as angstroms and converts them to nm.
_ANGSTROM_TO_NM = 0.1

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


def _position_frames(positions: np.ndarray) -> np.ndarray:
    frames = np.asarray(positions, dtype=np.float64)
    if frames.ndim == 3 and frames.shape[-1] == 3 and frames.shape[1] > 0:
        return frames
    if frames.ndim == 2 and frames.shape[-1] % 3 == 0 and frames.shape[-1] > 0:
        n_atoms = frames.shape[-1] // 3
        return frames.reshape(frames.shape[0], n_atoms, 3)
    raise ValueError("positions must have shape (n_frames, n_atoms, 3)")


def _reference_frame(reference: np.ndarray, n_atoms: int) -> np.ndarray:
    frame = np.asarray(reference, dtype=np.float64)
    if frame.shape == (n_atoms, 3):
        return frame
    if frame.shape == (n_atoms * 3,):
        return frame.reshape(n_atoms, 3)
    raise ValueError(f"reference must have shape ({n_atoms}, 3)")


def _mode_stack(mode_vectors: np.ndarray, n_atoms: int) -> np.ndarray:
    modes = np.asarray(mode_vectors, dtype=np.float64)
    if modes.ndim == 1:
        modes = modes.reshape(1, -1)
    if modes.shape[0] == 0:
        raise ValueError("mode_vectors must contain at least one mode")
    if modes.ndim == 3 and modes.shape[1:] == (n_atoms, 3):
        return modes
    if modes.ndim == 2 and modes.shape == (n_atoms, 3):
        return modes.reshape(1, n_atoms, 3)
    if modes.ndim == 2 and modes.shape[-1] == n_atoms * 3:
        return modes.reshape(modes.shape[0], n_atoms, 3)
    raise ValueError(f"mode_vectors must describe modes of {n_atoms} atoms")


def projection_standard_deviations(
    positions: np.ndarray,
    mode_vectors: np.ndarray,
    masses: np.ndarray,
    reference: np.ndarray,
    *,
    scale: float = 100.0,
) -> np.ndarray:
    """Standard deviation of each PLUMED mode projection, in nm.

    Displacements from ``reference`` are projected with the direction
    weights ``A_i = scale * u_i * sqrt(m_i)``. ``positions`` and
    ``reference`` are MDAnalysis coordinates in angstroms, already in
    one aligned frame. PLUMED converts the displacements to nanometres
    and normalizes each ``DIRECTION``, so these values match
    ``plumed driver``. The protocol uses half of each value as
    ``SIGMA``. ``scale`` must match the mode PDB.
    """
    frames = _position_frames(positions)
    if frames.shape[0] == 0:
        raise ValueError("positions must contain at least one frame")
    n_atoms = int(frames.shape[1])
    masses = np.asarray(masses, dtype=np.float64).reshape(-1)
    if masses.size != n_atoms:
        raise ValueError("masses length must match the number of atoms")
    origin = _reference_frame(reference, n_atoms)
    modes = _mode_stack(mode_vectors, n_atoms)
    directions = np.stack(
        [
            plumed_directions(mode, masses, scale=scale, n_atoms=n_atoms)
            for mode in modes
        ]
    )
    lengths = np.linalg.norm(directions, axis=(1, 2))
    if not np.all(np.isfinite(lengths)) or np.any(lengths == 0.0):
        raise ValueError("directions must be finite and nonzero")
    projections = np.einsum(
        "mai,fai->fm",
        directions / lengths[:, None, None],
        (frames - origin) * _ANGSTROM_TO_NM,
    )
    if not np.all(np.isfinite(projections)):
        raise ValueError("projections must be finite")
    return np.std(projections, axis=0)


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


def _deposition_steps(timestep: float, deposition: float) -> int:
    """Timesteps between Gaussians. Both times are in picoseconds."""
    if timestep <= 0.0 or deposition <= 0.0:
        raise ValueError("timestep and deposition must be positive")
    steps = deposition / timestep
    pace = int(round(steps))
    if pace < 1 or abs(steps - pace) > 1e-6 * max(1.0, abs(steps)):
        raise ValueError("deposition must be a whole number of timesteps")
    return pace


def write_plumed_metad_input(  # pylint: disable=too-many-arguments
    path: str,
    sigmas: Sequence[float],
    *,
    timestep: float = 0.002,
    deposition: float = 1.0,
    height: float = 0.1,
    biasfactor: float = 10.0,
    temperature: float = 300.0,
    reference: str = "plumed-mode-input.pdb",
    hills: str = "plumed-mode-metadyn.hills",
    output: str = "plumed-mode-metadyn.out",
) -> None:
    """Write a well-tempered metadynamics script on the mode projections.

    ``sigmas`` is one positive Gaussian width per mode. The FRESEAN
    protocol uses half of ``projection_standard_deviations``.
    ``timestep`` and ``deposition`` are in picoseconds. ``PACE`` is
    ``deposition / timestep`` and defaults to 1 ps. Pass a longer
    deposition when the collective variable autocorrelation decays
    more slowly.

    Height, bias factor, and temperature default to 0.1 kJ/mol, 10,
    and 300 K.
    """
    widths = [float(value) for value in sigmas]
    if not widths or any(not np.isfinite(value) or value <= 0.0 for value in widths):
        raise ValueError("sigmas must be positive and finite")
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError("height must be positive and finite")
    if not np.isfinite(biasfactor) or biasfactor < 1.0:
        raise ValueError("biasfactor must be at least 1")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")

    pace = _deposition_steps(timestep, deposition)
    arguments = ",".join(f"pca.eig-{index}" for index in range(1, len(widths) + 1))
    text = (
        f"PCAVARS REFERENCE={reference} TYPE=OPTIMAL LABEL=pca\n"
        "\n"
        "METAD ...\n"
        "LABEL=metad\n"
        f"ARG={arguments}\n"
        f"PACE={pace}\n"
        f"HEIGHT={height:.12g}\n"
        f"BIASFACTOR={biasfactor:.12g}\n"
        f"SIGMA={','.join(f'{value:.12g}' for value in widths)}\n"
        f"FILE={hills}\n"
        f"TEMP={temperature:.12g}\n"
        "... METAD\n"
        "\n"
        "# monitor the two variables and the metadynamics bias potential\n"
        f"PRINT ARG={arguments},pca.residual,metad.bias "
        f"FILE={output} STRIDE={pace}\n"
    )
    with open(path, "w", encoding="utf-8") as writer:
        writer.write(text)

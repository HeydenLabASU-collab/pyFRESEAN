from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple, Union

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

from pyfresean.exceptions import MissingBeadMassWarning, TopologyFormatWarning

if TYPE_CHECKING:
    from MDAnalysis.core.groups import AtomGroup


@dataclass
class CoarseGrainMap:
    """Maps coarse-grained beads to atoms in a source atom group."""

    bead_names: List[str]
    bead_masses: np.ndarray
    resindices: np.ndarray
    resnames: List[str]
    bead_types: List[str]
    atom_indices: List[np.ndarray]
    n_constraints: float = 6.0

    @property
    def n_beads(self) -> int:
        return len(self.bead_names)


class CoarseGrain:
    """Coarse-grain an all-atom atom group to backbone / sidechain COM beads."""

    BACKBONE_ATOM_NAMES = frozenset(
        {"CA", "C", "O", "N", "H", "HA", "H1", "H2", "H3", "OC1", "OC2"}
        # Current naming scheme is AMBER
        # Support for CHARMM terminal/backbone names (HT1–HT3, HN) not included yet; see issue note.
    )
    BACKBONE_ONLY_RESIDUES = frozenset({"ACE", "NME", "GLY"})
    ELEMENT_MASSES = {
        "H": 1.0080,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "S": 32.070,
        "P": 30.973762,
    }

    def __init__(self, atomgroup: AtomGroup, n_constraints: float = 6.0):
        self.atomgroup = atomgroup
        self._n_constraints = n_constraints
        self._mapping: Optional[CoarseGrainMap] = None
        self._topology_universe: Optional[mda.Universe] = None
        self._scratch_universe: Optional[mda.Universe] = None
        self._memory_universe: Optional[mda.Universe] = None
        self._memory_n_frames: int = 0

    def _ensure_mapping(self) -> CoarseGrainMap:
        if self._mapping is None:
            self._mapping = self._build_mapping(self.atomgroup, self._n_constraints)
        return self._mapping

    @property
    def mapping(self) -> CoarseGrainMap:
        """Bead-to-atom mapping; built on first access."""
        return self._ensure_mapping()

    @classmethod
    def from_atomgroup(
        cls,
        atomgroup: AtomGroup,
        n_constraints: float = 6.0,
    ) -> CoarseGrain:
        cg = cls(atomgroup=atomgroup, n_constraints=n_constraints)
        cg._ensure_mapping()
        return cg

    @classmethod
    def _warn_topology_format(cls, topology_path: str | Path) -> None:
        if cls._uses_top_file(topology_path):
            return
        warnings.warn(
            "output_top uses a format that may not store bead masses on reload; "
            "use .top or .itp to preserve masses.",
            TopologyFormatWarning,
            stacklevel=4,
        )

    @classmethod
    def cg_universe(
        cls,
        topology: Union[str, Path],
        trajectory: Union[str, Path, None] = None,
        select: str = "all",
        n_constraints: float = 6.0,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        frames: Optional[Iterable[int]] = None,
        output_traj: Optional[str] = None,
        output_top: Optional[str] = None,
        in_memory: bool = True,
    ) -> Tuple[CoarseGrain, mda.Universe]:
        """Load an all-atom trajectory and return a coarse-grained universe.

        Parameters
        ----------
        topology
            All-atom topology file (``.tpr``, ``.pdb``, ``.gro``, ...).
        trajectory
            All-atom trajectory file. Optional when ``topology`` includes frames.
        select
            Atom selection applied before coarse-graining.
        n_constraints
            Number of holonomic constraints removed for FRESEAN (default 6).
        output_traj
            If set, write the CG trajectory (``.trr``, ``.xtc``, ...).
        output_top
            If set, write the CG topology. ``.top`` / ``.itp`` store bead masses;
            other formats trigger a :class:`~pyfresean.exceptions.TopologyFormatWarning`.
        in_memory
            If ``False``, stream to ``output_top`` and ``output_traj`` without
            holding the full CG trajectory in RAM (both outputs required).

        Returns
        -------
        cg
            :class:`CoarseGrain` built from the selected atoms (for mapping,
            back-mapping, and PLUMED export).
        universe
            Coarse-grained MDAnalysis universe.
        """
        if trajectory is None:
            u_aa = mda.Universe(str(topology))
        else:
            u_aa = mda.Universe(str(topology), str(trajectory))
        atomgroup = u_aa.select_atoms(select)
        if len(atomgroup) == 0:
            raise ValueError(f"selection {select!r} matched no atoms")
        cg = cls.from_atomgroup(atomgroup, n_constraints=n_constraints)
        if output_top is not None:
            cls._warn_topology_format(output_top)
        u_cg = cg.build_universe(
            start=start,
            stop=stop,
            step=step,
            frames=frames,
            output_traj=output_traj,
            output_top=output_top,
            in_memory=in_memory,
        )
        return cg, u_cg

    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @classmethod
    def _is_backbone_only(cls, resname: str) -> bool:
        return cls._strip(resname) in cls.BACKBONE_ONLY_RESIDUES

    @classmethod
    def _is_backbone_atom(cls, atom_name: str) -> bool:
        return cls._strip(atom_name) in cls.BACKBONE_ATOM_NAMES

    @classmethod
    def _guess_mass_from_name(cls, atom_name: str) -> float:
        """Guess atomic mass from the first character of ``atom_name``."""
        element = cls._strip(atom_name)[0].upper()
        try:
            return cls.ELEMENT_MASSES[element]
        except KeyError as exc:
            raise ValueError(
                f"cannot guess mass for atom name {atom_name!r}; "
                f"set masses on the atom group or add the element to ELEMENT_MASSES"
            ) from exc

    def _atom_masses(self) -> np.ndarray:
        """Return per-atom masses, guessing from atom names when missing."""
        return self._resolved_masses(self.atomgroup)

    @classmethod
    def _resolved_masses(cls, atomgroup: AtomGroup) -> np.ndarray:
        masses = np.asarray(atomgroup.masses, dtype=np.float64).copy()
        missing = ~np.isfinite(masses) | (masses <= 0.0)
        if np.any(missing):
            for idx in np.where(missing)[0]:
                masses[idx] = cls._guess_mass_from_name(atomgroup.names[idx])
        return masses

    @classmethod
    def _build_mapping(
        cls,
        atomgroup: AtomGroup,
        n_constraints: float = 6.0,
    ) -> CoarseGrainMap:
        masses = cls._resolved_masses(atomgroup)
        bead_names: List[str] = []
        bead_masses: List[float] = []
        resindices: List[int] = []
        resnames: List[str] = []
        bead_types: List[str] = []
        atom_indices: List[np.ndarray] = []

        for residue in atomgroup.residues:
            res_atoms = residue.atoms & atomgroup
            if len(res_atoms) == 0:
                continue

            back_idx: List[int] = []
            side_idx: List[int] = []
            backbone_only = cls._is_backbone_only(residue.resname)

            for atom in res_atoms:
                local_idx = int(np.where(atomgroup.indices == atom.index)[0][0])
                if backbone_only or cls._is_backbone_atom(atom.name):
                    back_idx.append(local_idx)
                else:
                    side_idx.append(local_idx)

            if back_idx:
                bead_names.append("BACK")
                bead_masses.append(float(np.sum(masses[back_idx])))
                resindices.append(int(residue.resindex))
                resnames.append(cls._strip(residue.resname))
                bead_types.append("BACK")
                atom_indices.append(np.asarray(back_idx, dtype=np.int64))

            if side_idx:
                bead_names.append("SIDE")
                bead_masses.append(float(np.sum(masses[side_idx])))
                resindices.append(int(residue.resindex))
                resnames.append(cls._strip(residue.resname))
                bead_types.append("SIDE")
                atom_indices.append(np.asarray(side_idx, dtype=np.int64))

        if not bead_names:
            raise ValueError("atom group produced no coarse-grained beads")

        return CoarseGrainMap(
            bead_names=bead_names,
            bead_masses=np.asarray(bead_masses, dtype=np.float64),
            resindices=np.asarray(resindices, dtype=np.int32),
            resnames=resnames,
            bead_types=bead_types,
            atom_indices=atom_indices,
            n_constraints=n_constraints,
        )

    @staticmethod
    def _weighted_com(
        positions: np.ndarray,
        masses: Sequence[float],
        indices: np.ndarray,
    ) -> np.ndarray:
        weights = np.asarray(masses, dtype=np.float64)[indices]
        coords = positions[indices]
        total = np.sum(weights)
        if total <= 0:
            raise ValueError("bead mass must be positive")
        return np.average(coords, axis=0, weights=weights)

    def map_positions(self, positions: np.ndarray) -> np.ndarray:
        mapping = self._ensure_mapping()
        masses = self._atom_masses()
        return np.asarray(
            [
                self._weighted_com(positions, masses, indices)
                for indices in mapping.atom_indices
            ],
            dtype=np.float64,
        )

    def map_velocities(self, velocities: np.ndarray) -> np.ndarray:
        return self.map_positions(velocities)

    def _create_cg_universe(self, n_frames: int) -> mda.Universe:
        """Build a new CG universe shell (topology + optional MemoryReader)."""
        mapping = self._ensure_mapping()
        n_beads = mapping.n_beads
        n_residues = int(np.max(mapping.resindices)) + 1

        residue_bead_counts = np.bincount(mapping.resindices, minlength=n_residues)
        atom_resindex = np.repeat(
            np.arange(n_residues, dtype=np.int32),
            residue_bead_counts,
        )

        u = mda.Universe.empty(
            n_atoms=n_beads,
            n_residues=n_residues,
            n_segments=1,
            atom_resindex=atom_resindex,
            residue_segindex=np.zeros(n_residues, dtype=np.int32),
            trajectory=n_frames > 0,
            velocities=n_frames > 0,
            n_frames=max(n_frames, 1) if n_frames > 0 else 1,
        )
        u.add_TopologyAttr("masses")
        u.add_TopologyAttr("names")
        u.add_TopologyAttr("resnames")
        u.add_TopologyAttr("resids")

        u.atoms.masses = mapping.bead_masses
        u.atoms.names = np.array(mapping.bead_names, dtype=object)
        u.residues.resnames = np.array(
            [
                mapping.resnames[int(np.where(mapping.resindices == i)[0][0])]
                for i in range(n_residues)
            ],
            dtype=object,
        )
        u.residues.resids = np.arange(1, n_residues + 1)

        if n_frames > 0:
            positions = np.zeros((n_frames, n_beads, 3), dtype=np.float32)
            velocities = np.zeros((n_frames, n_beads, 3), dtype=np.float32)
            u.trajectory = MemoryReader(positions, velocities=velocities)
        return u

    def empty_cg_universe(self, n_frames: int = 0) -> mda.Universe:
        """Return a CG universe with bead topology and optional trajectory slots.

        Topology-only (``n_frames=0``), single-frame scratch (``n_frames=1``),
        and repeated in-memory builds with the same frame count reuse cached
        universes on this :class:`CoarseGrain` instance.
        """
        self._ensure_mapping()
        if n_frames == 0:
            if self._topology_universe is None:
                self._topology_universe = self._create_cg_universe(n_frames=0)
            return self._topology_universe
        if n_frames == 1:
            if self._scratch_universe is None:
                self._scratch_universe = self._create_cg_universe(n_frames=1)
            return self._scratch_universe
        if self._memory_universe is not None and self._memory_n_frames == n_frames:
            return self._memory_universe
        self._memory_universe = self._create_cg_universe(n_frames=n_frames)
        self._memory_n_frames = n_frames
        return self._memory_universe

    def _bead_masses_trusted(self, universe: mda.Universe) -> bool:
        """Return whether ``universe`` already has usable CG bead masses."""
        mapping = self._ensure_mapping()
        masses = np.asarray(universe.atoms.masses, dtype=np.float64)
        expected = np.asarray(mapping.bead_masses, dtype=np.float64)
        if masses.shape != expected.shape:
            return False
        if not np.all(np.isfinite(masses)) or not np.all(masses > 0.0):
            return False
        return bool(np.allclose(masses, expected, rtol=1e-4, atol=1e-4))

    def _restore_bead_masses(self, universe: mda.Universe) -> None:
        """Set bead masses from the mapping when the topology does not provide them."""
        if self._bead_masses_trusted(universe):
            return
        mapping = self._ensure_mapping()
        warnings.warn(
            "CG bead masses are missing or unreliable on the loaded topology; "
            "restoring them from the coarse-grain mapping.",
            MissingBeadMassWarning,
            stacklevel=3,
        )
        universe.atoms.masses = mapping.bead_masses

    @staticmethod
    def _topology_coord_path(topology_path: str | Path) -> Path:
        return Path(topology_path).with_suffix(".gro")

    @staticmethod
    def _uses_top_file(topology_path: str | Path) -> bool:
        return Path(topology_path).suffix.lower() in {".top", ".itp"}

    def _write_top_file(self, universe: mda.Universe, path: str) -> None:
        """Write a minimal ``.top`` file with explicit bead masses."""
        mapping = self._ensure_mapping()
        lines = [
            ";",
            "; PyFRESEAN coarse-grained topology",
            ";",
            "",
            "[ defaults ]",
            "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ",
            "  1   2   no   1.0   1.0",
            "",
            "[ atomtypes ]",
            "; name bond-type mass charge ptype sigma epsilon",
            "  CG   1   1.0   0.0   A   0.0   0.0",
            "",
            "[ moleculetype ]",
            "; name nrexcl",
            "  CGmol 1",
            "",
            "[ atoms ]",
            "; nr type resnr residue atom cgnr charge mass",
        ]
        for bead_idx in range(mapping.n_beads):
            resnr = int(mapping.resindices[bead_idx]) + 1
            resname = mapping.resnames[bead_idx]
            atomname = mapping.bead_names[bead_idx]
            mass = float(mapping.bead_masses[bead_idx])
            lines.append(
                f"  {bead_idx + 1:5d} CG {resnr:5d} {resname:>5} {atomname:>4} "
                f"{bead_idx + 1:5d} 0.0 {mass:12.6f}"
            )
        lines.extend(
            [
                "",
                "[ system ]",
                "CG",
                "",
                "[ molecules ]",
                "CGmol 1",
                "",
            ]
        )
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    def _load_cg_universe_from_disk(
        self,
        output_top: str,
        output_traj: str,
    ) -> mda.Universe:
        if self._uses_top_file(output_top):
            gro_path = self._topology_coord_path(output_top)
            u = mda.Universe(
                output_top,
                str(gro_path),
                output_traj,
                topology_format="ITP",
            )
            self._restore_bead_masses(u)
            return u
        u = mda.Universe(output_top, output_traj)
        self._restore_bead_masses(u)
        return u

    def write_topology(self, universe: mda.Universe, path: str) -> None:
        """Write CG topology (``.top``, ``.pdb``, ``.gro``, ...) from the first frame.

        ``.top`` / ``.itp`` also write a companion ``.gro`` with first-frame positions.
        """
        if len(universe.trajectory):
            universe.trajectory[0]
        if self._uses_top_file(path):
            gro_path = self._topology_coord_path(path)
            universe.atoms.write(str(gro_path))
            self._write_top_file(universe, path)
            return
        universe.atoms.write(path)

    def write_trajectory(
        self,
        universe: mda.Universe,
        path: str,
        dimensions: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> None:
        """Write CG trajectory (``.trr``, ``.xtc``, ...) — format from extension."""
        with mda.Writer(path, n_atoms=universe.atoms.n_atoms) as writer:
            for frame_idx, _ts in enumerate(universe.trajectory):
                if dimensions is not None and dimensions[frame_idx] is not None:
                    universe.trajectory.ts.dimensions = dimensions[frame_idx]
                writer.write(universe.atoms)

    def _resolve_frame_list(
        self,
        trajectory,
        start: Optional[int],
        stop: Optional[int],
        step: Optional[int],
        frames: Optional[Iterable[int]],
    ) -> list[int]:
        if frames is not None:
            return list(frames)
        return list(range(len(trajectory)))[slice(start, stop, step)]

    def _fill_scratch_frame(
        self,
        scratch: mda.Universe,
        positions: np.ndarray,
        velocities: Optional[np.ndarray],
        dimensions: Optional[np.ndarray],
    ) -> None:
        ts = scratch.trajectory[0]
        ts.positions = np.asarray(positions, dtype=np.float32)
        if velocities is not None:
            ts.velocities = np.asarray(velocities, dtype=np.float32)
        if dimensions is not None:
            ts.dimensions = dimensions

    def _cg_universe_to_disk(
        self,
        frame_list: list[int],
        output_top: str,
        output_traj: str,
    ) -> mda.Universe:
        trajectory = self.atomgroup.universe.trajectory
        ag = self.atomgroup
        scratch = self.empty_cg_universe(n_frames=1)
        has_velocities = None

        with mda.Writer(output_traj, n_atoms=scratch.atoms.n_atoms) as writer:
            for frame_idx, frame in enumerate(frame_list):
                trajectory[frame]
                bead_pos = self.map_positions(ag.positions)
                if has_velocities is None:
                    has_velocities = trajectory.ts.has_velocities
                bead_vel = (
                    self.map_velocities(ag.velocities) if has_velocities else None
                )
                dims = trajectory.ts.dimensions
                box = None if dims is None else np.asarray(dims, dtype=np.float32)
                self._fill_scratch_frame(scratch, bead_pos, bead_vel, box)
                if frame_idx == 0:
                    if self._uses_top_file(output_top):
                        gro_path = self._topology_coord_path(output_top)
                        scratch.atoms.write(str(gro_path))
                        self._write_top_file(scratch, output_top)
                    else:
                        self._warn_topology_format(output_top)
                        scratch.atoms.write(output_top)
                writer.write(scratch.atoms)

        return self._load_cg_universe_from_disk(output_top, output_traj)

    def build_universe(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        frames: Optional[Iterable[int]] = None,
        output_traj: Optional[str] = None,
        output_top: Optional[str] = None,
        in_memory: bool = True,
    ) -> mda.Universe:
        """Build a CG universe from the source atom group already on this instance.

        Prefer :meth:`cg_universe` when starting from all-atom file paths.
        """
        self._ensure_mapping()
        trajectory = self.atomgroup.universe.trajectory
        ag = self.atomgroup
        frame_list = self._resolve_frame_list(trajectory, start, stop, step, frames)

        if not frame_list:
            return self.empty_cg_universe(n_frames=0)

        if not in_memory:
            if output_traj is None or output_top is None:
                raise ValueError(
                    "in_memory=False requires both output_traj and output_top"
                )
            return self._cg_universe_to_disk(frame_list, output_top, output_traj)

        positions = []
        velocities = []
        dimensions_list: list[Optional[np.ndarray]] = []
        has_velocities = None

        for frame in frame_list:
            trajectory[frame]
            positions.append(self.map_positions(ag.positions))
            if output_traj is not None:
                dims = trajectory.ts.dimensions
                dimensions_list.append(
                    None if dims is None else np.asarray(dims, dtype=np.float32)
                )
            if has_velocities is None:
                has_velocities = trajectory.ts.has_velocities
            if has_velocities:
                velocities.append(self.map_velocities(ag.velocities))

        n_frames = len(positions)
        u = self.empty_cg_universe(n_frames=n_frames)
        pos_array = np.asarray(positions, dtype=np.float32)
        for frame_idx, ts in enumerate(u.trajectory):
            ts.positions = pos_array[frame_idx]
            if has_velocities:
                ts.velocities = np.asarray(velocities[frame_idx], dtype=np.float32)

        if output_top is not None:
            self._warn_topology_format(output_top)
            self.write_topology(u, output_top)
        if output_traj is not None:
            self.write_trajectory(u, output_traj, dimensions=dimensions_list)

        return u

    def backmap_modes(self, cg_vectors: np.ndarray) -> np.ndarray:
        """Distribute CG mode vectors onto all-atom coordinates.

        Each bead eigenvector component ``u_b`` (in sqrt(M_b)*V space) is
        expanded onto atoms as ``u_i = u_b * sqrt(m_i / M_b)``. Velocities are
        uniform within a bead: ``u_i / sqrt(m_i) = u_b / sqrt(M_b)``.
        """
        mapping = self._ensure_mapping()
        if cg_vectors.ndim == 1:
            cg_vectors = cg_vectors.reshape(mapping.n_beads, 3)
        if cg_vectors.shape != (mapping.n_beads, 3):
            raise ValueError(
                f"expected cg_vectors shape ({mapping.n_beads}, 3), got {cg_vectors.shape}"
            )

        masses = self._atom_masses()
        n_atoms = self.atomgroup.n_atoms
        aa_vectors = np.zeros((n_atoms, 3), dtype=np.float64)
        for bead_idx, indices in enumerate(mapping.atom_indices):
            if len(indices) == 0:
                continue
            bead_mass = float(np.sum(masses[indices]))
            if bead_mass <= 0.0:
                raise ValueError(f"bead {bead_idx} has non-positive total mass")
            scale = np.sqrt(masses[indices] / bead_mass)[:, np.newaxis]
            aa_vectors[indices] = cg_vectors[bead_idx] * scale
        return aa_vectors

    def scale_plumed_directions(
        self,
        aa_vectors: np.ndarray,
        scale: float = 100.0,
    ) -> np.ndarray:
        """Scale back-mapped mode vectors for PLUMED ``DIRECTION`` PDBs.

        Uses ``scale * aa_vectors * sqrt(masses)``.
        """
        aa_vectors = np.asarray(aa_vectors, dtype=np.float64)
        n_atoms = self.atomgroup.n_atoms
        if aa_vectors.ndim == 1:
            aa_vectors = aa_vectors.reshape(n_atoms, 3)
        if aa_vectors.shape != (n_atoms, 3):
            raise ValueError(
                f"expected aa_vectors shape ({n_atoms}, 3), got {aa_vectors.shape}"
            )

        masses = self._atom_masses()
        return scale * aa_vectors * np.sqrt(masses)[:, np.newaxis]

    @staticmethod
    def _pdb_record_name(atom) -> str:
        if hasattr(atom, "record_type"):
            return str(atom.record_type)
        return "ATOM"

    @staticmethod
    def _pdb_chain_id(atom) -> str:
        if hasattr(atom, "chainID") and atom.chainID not in (None, ""):
            return str(atom.chainID)[:1]
        if hasattr(atom, "segid") and atom.segid not in (None, ""):
            return str(atom.segid)[:1]
        return " "

    def _format_pdb_atom_line(
        self,
        atom,
        coords: Sequence[float],
    ) -> str:
        x, y, z = coords
        return (
            f"{self._pdb_record_name(atom):<6}{int(atom.index) + 1:5d} "
            f"{atom.name:>4s} {atom.resname:>3s}{self._pdb_chain_id(atom):1s}"
            f"{int(atom.resid):4d}    {float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
            f"  1.00  0.00\n"
        )

    def write_plumed_direction_pdb(
        self,
        aa_vectors: np.ndarray,
        path: str,
        scale: float = 100.0,
    ) -> None:
        """Write one PLUMED ``DIRECTION`` PDB for a back-mapped all-atom mode."""
        directions = self.scale_plumed_directions(aa_vectors, scale=scale)
        with open(path, "w", encoding="utf-8") as writer:
            writer.write("REMARK TYPE=DIRECTION\n")
            for atom, coords in zip(self.atomgroup, directions):
                writer.write(self._format_pdb_atom_line(atom, coords))
            writer.write("END\n")

    def write_plumed_mode_input(
        self,
        aa_modes: Sequence[np.ndarray],
        path: str,
        scale: float = 100.0,
        include_reference: bool = True,
    ) -> None:
        """Write a combined PLUMED reference + mode ``DIRECTION`` PDB.

        Same layout as ``ref.pdb`` followed by one ``evec_*_aa_scaled.pdb`` per mode.
        """
        if len(aa_modes) == 0:
            raise ValueError("aa_modes must contain at least one mode")

        with open(path, "w", encoding="utf-8") as writer:
            if include_reference:
                writer.write("REMARK TYPE=OPTIMAL\n")
                for atom in self.atomgroup:
                    writer.write(
                        self._format_pdb_atom_line(atom, atom.position)
                    )
                writer.write("END\n")

            for aa_vectors in aa_modes:
                directions = self.scale_plumed_directions(aa_vectors, scale=scale)
                writer.write("REMARK TYPE=DIRECTION\n")
                for atom, coords in zip(self.atomgroup, directions):
                    writer.write(self._format_pdb_atom_line(atom, coords))
                writer.write("END\n")

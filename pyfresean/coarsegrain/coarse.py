from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

from pyfresean.benchmark_keys import (
    BENCH_T_ASSEMBLE_UNIVERSE,
    BENCH_T_FRAME_PROCESSING,
    BENCH_T_MAPPING,
    BENCH_T_WRITE_OUTPUTS,
    finalize_cg_benchmark,
    new_cg_benchmark_timings,
)
from pyfresean.exceptions import MissingBeadMassWarning, TopologyFormatWarning

if TYPE_CHECKING:
    from MDAnalysis.core.groups import AtomGroup

SUPPORTED_CG_METHODS = [
    "backbone-sidechain",
    "martini",
]

UniverseSource = Union[
    mda.Universe,
    str,
    Path,
    Tuple[Union[str, Path], Optional[Union[str, Path]]],
    Sequence[Union[str, Path]],
]


@dataclass
class CoarseGrainMap:
    """Maps coarse-grained beads to atoms in a source atom group."""

    bead_names: List[str]
    bead_masses: np.ndarray
    resindices: np.ndarray
    resnames: List[str]
    bead_types: List[str]
    atom_indices: List[np.ndarray]
    n_constraints: float = 0.0
    reference_frame: int = 0

    @property
    def n_beads(self) -> int:
        return len(self.bead_names)


MapSource = Union[CoarseGrainMap, str, Path]


class CoarseGrain:
    """Coarse-grain an all-atom atom group to COM beads.

    Parameters
    ----------
    atomgroup
        Source atoms to coarse-grain.
    n_constraints
        Number of constraints applied to atoms in the selection (default 0).
    method
        Coarse-graining scheme. ``"backbone-sidechain"`` maps each residue to a
        backbone COM bead and (except GLY/ACE/NME) a sidechain COM bead.
        ``"martini"`` follows Martini mapping (not yet implemented).
    """

    CG_METHODS = SUPPORTED_CG_METHODS

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

    def __init__(
        self,
        atomgroup: AtomGroup,
        n_constraints: float = 0.0,
        method: str = "backbone-sidechain",
    ):
        self.atomgroup = atomgroup
        self._n_constraints = n_constraints
        self._method = self._normalize_method(method)
        self._mapping: Optional[CoarseGrainMap] = None
        self._topology_universe: Optional[mda.Universe] = None
        self._scratch_universe: Optional[mda.Universe] = None
        self._memory_universe: Optional[mda.Universe] = None
        self._memory_n_frames: int = 0
        self._reference_frame: int = 0
        self.benchmark: Optional[dict[str, float]] = None

    def _ensure_mapping(
        self, timings: Optional[dict[str, float]] = None
    ) -> CoarseGrainMap:
        if self._mapping is None:
            if timings is not None:
                t_mapping = time.perf_counter()
            trajectory = self.atomgroup.universe.trajectory
            if trajectory is not None:
                trajectory[self._reference_frame]
            self._mapping = self._build_mapping(
                self.atomgroup,
                self._n_constraints,
                self._method,
            )
            self._mapping.reference_frame = self._reference_frame
            if timings is not None:
                timings[BENCH_T_MAPPING] += time.perf_counter() - t_mapping
        return self._mapping

    @property
    def method(self) -> str:
        """Coarse-graining scheme used for this instance."""
        return self._method

    @property
    def reference_frame(self) -> int:
        """Trajectory frame used to build orientational reference data."""
        return self._reference_frame

    @staticmethod
    def _resolve_universe(source: UniverseSource) -> mda.Universe:
        """Build an MDAnalysis universe from a universe, path, or ``(top, traj)`` pair."""
        if isinstance(source, mda.Universe):
            return source
        if isinstance(source, (str, Path)):
            return mda.Universe(str(source))
        if isinstance(source, (tuple, list)):
            if not source or len(source) > 2:
                raise ValueError(
                    "expected a topology path or (topology, trajectory) pair, "
                    f"got sequence of length {len(source)}"
                )
            topology = str(source[0])
            if len(source) == 2 and source[1] is not None:
                return mda.Universe(topology, str(source[1]))
            return mda.Universe(topology)
        raise TypeError(
            "expected an MDAnalysis Universe, file path, or (topology, trajectory) pair; "
            f"got {type(source).__name__}"
        )

    def _apply_cg_map(self, cg_map: MapSource) -> None:
        if isinstance(cg_map, CoarseGrainMap):
            self._mapping = cg_map
            self._reference_frame = cg_map.reference_frame
            self._n_constraints = cg_map.n_constraints
            return
        mapping, method = self.load_mapping(cg_map)
        self._mapping = mapping
        self._method = method
        self._reference_frame = mapping.reference_frame
        self._n_constraints = mapping.n_constraints

    @classmethod
    def _normalize_method(cls, method: str) -> str:
        normalized = method.strip().lower().replace("_", "-")
        if normalized not in cls.CG_METHODS:
            supported = ", ".join(cls.CG_METHODS)
            raise ValueError(
                f"unsupported coarse-graining method {method!r}; "
                f"choose one of: {supported}"
            )
        return normalized

    @property
    def mapping(self) -> CoarseGrainMap:
        """Bead-to-atom mapping; built on first access."""
        return self._ensure_mapping()

    @classmethod
    def from_atomgroup(
        cls,
        atomgroup: AtomGroup,
        n_constraints: float = 0.0,
        method: str = "backbone-sidechain",
        reference: int = 0,
        timings: Optional[dict[str, float]] = None,
    ) -> CoarseGrain:
        trajectory = atomgroup.universe.trajectory
        if trajectory is not None:
            if reference < 0 or reference >= len(trajectory):
                raise ValueError(
                    f"reference frame {reference} out of range for trajectory "
                    f"with {len(trajectory)} frames"
                )
            trajectory[reference]
        cg = cls(
            atomgroup=atomgroup, n_constraints=n_constraints, method=method
        )
        cg._reference_frame = reference
        cg._ensure_mapping(timings=timings)
        return cg

    @classmethod
    def from_cg_map(
        cls,
        atomgroup: AtomGroup,
        map_path: Union[str, Path],
    ) -> CoarseGrain:
        """Build a :class:`CoarseGrain` from a saved mapping file."""
        mapping, method = cls.load_mapping(map_path)
        cg = cls(
            atomgroup=atomgroup,
            n_constraints=mapping.n_constraints,
            method=method,
        )
        cg._reference_frame = mapping.reference_frame
        cg._mapping = mapping
        return cg

    @staticmethod
    def _mapping_to_npz_dict(
        mapping: CoarseGrainMap,
        method: str,
    ) -> dict[str, np.ndarray]:
        data: dict[str, np.ndarray] = {
            "method": np.array(method),
            "n_constraints": np.array(mapping.n_constraints),
            "reference_frame": np.array(mapping.reference_frame),
            "bead_masses": mapping.bead_masses,
            "resindices": mapping.resindices,
            "bead_names": np.array(mapping.bead_names, dtype=object),
            "resnames": np.array(mapping.resnames, dtype=object),
            "bead_types": np.array(mapping.bead_types, dtype=object),
            "n_beads": np.array(mapping.n_beads),
        }
        for bead_idx in range(mapping.n_beads):
            data[f"atom_indices_{bead_idx}"] = mapping.atom_indices[bead_idx]
        return data

    @classmethod
    def load_mapping(
        cls,
        map_path: Union[str, Path],
    ) -> Tuple[CoarseGrainMap, str]:
        """Load a :class:`CoarseGrainMap` and method name from an ``.npz`` file."""
        with np.load(str(map_path), allow_pickle=True) as archive:
            n_beads = int(archive["n_beads"])
            method = str(archive["method"])
            bead_names = list(archive["bead_names"])
            resnames = list(archive["resnames"])
            bead_types = list(archive["bead_types"])
            atom_indices: List[np.ndarray] = []
            for bead_idx in range(n_beads):
                atom_indices.append(
                    np.asarray(
                        archive[f"atom_indices_{bead_idx}"], dtype=np.int64
                    )
                )
            mapping = CoarseGrainMap(
                bead_names=bead_names,
                bead_masses=np.asarray(
                    archive["bead_masses"], dtype=np.float64
                ),
                resindices=np.asarray(archive["resindices"], dtype=np.int32),
                resnames=resnames,
                bead_types=bead_types,
                atom_indices=atom_indices,
                n_constraints=float(archive["n_constraints"]),
                reference_frame=int(archive["reference_frame"]),
            )
        return mapping, method

    def save_mapping(self, map_path: Union[str, Path]) -> CoarseGrainMap:
        """Write :attr:`mapping` to an ``.npz`` file and return it."""
        mapping = self._ensure_mapping()
        data = self._mapping_to_npz_dict(mapping, self.method)
        np.savez_compressed(str(map_path), **data)
        return mapping

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
        aa: UniverseSource,
        select: str = "all",
        n_constraints: float = 0.0,
        method: str = "backbone-sidechain",
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        frames: Optional[Iterable[int]] = None,
        output_cg_trajectory: Optional[str] = None,
        output_cg_topology: Optional[str] = None,
        in_memory: bool = True,
        reference: int = 0,
        output_cg_map: Optional[str] = None,
        benchmark: bool = False,
    ) -> Tuple[CoarseGrain, mda.Universe]:
        """Build a coarse-grained universe from all-atom input.

        Parameters
        ----------
        aa
            All-atom input as an MDAnalysis :class:`~MDAnalysis.core.universe.Universe`,
            a topology file path, or ``(topology, trajectory)``.
        select
            Atom selection applied before coarse-graining.
        n_constraints
            Stored on the mapping for downstream FRESEAN (default 0).
        method
            Coarse-graining scheme (``"backbone-sidechain"`` or ``"martini"``).
        output_cg_trajectory
            If set, write the CG trajectory (``.trr``, ``.xtc``, ...).
        output_cg_topology
            If set, write the CG topology. ``.top`` / ``.itp`` store bead masses;
            other formats trigger a :class:`~pyfresean.exceptions.TopologyFormatWarning`.
        in_memory
            If ``False``, stream to ``output_cg_topology`` and ``output_cg_trajectory``
            without holding the full CG trajectory in RAM (both outputs required).
        reference
            Trajectory frame index stored on the mapping (default first frame).
            Bead membership does not depend on coordinates.
        output_cg_map
            If set, write :attr:`CoarseGrain.mapping` to an ``.npz`` file.
        benchmark
            If ``True``, record wall times for the major coarse-graining phases
            on :attr:`CoarseGrain.benchmark`.

        Returns
        -------
        cg
            :class:`CoarseGrain` built from the selected atoms (for mapping,
            back-mapping, and PLUMED export).
        universe
            Coarse-grained MDAnalysis universe.
        """
        u_aa = cls._resolve_universe(aa)

        atomgroup = u_aa.select_atoms(select)
        if len(atomgroup) == 0:
            raise ValueError(f"selection {select!r} matched no atoms")
        timings = new_cg_benchmark_timings() if benchmark else None
        cg = cls.from_atomgroup(
            atomgroup,
            n_constraints=n_constraints,
            method=method,
            reference=reference,
            timings=timings,
        )
        if output_cg_topology is not None:
            cls._warn_topology_format(output_cg_topology)
        u_cg = cg.build_universe(
            start=start,
            stop=stop,
            step=step,
            frames=frames,
            output_cg_trajectory=output_cg_trajectory,
            output_cg_topology=output_cg_topology,
            in_memory=in_memory,
            output_cg_map=output_cg_map,
            benchmark=benchmark,
            timings=timings,
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
        n_constraints: float = 0.0,
        method: str = "backbone-sidechain",
    ) -> CoarseGrainMap:
        method = cls._normalize_method(method)
        if method == "backbone-sidechain":
            return cls._build_backbone_sidechain_mapping(
                atomgroup, n_constraints
            )
        if method == "martini":
            return cls._build_martini_mapping(atomgroup, n_constraints)
        supported = ", ".join(cls.CG_METHODS)
        raise ValueError(
            f"unsupported coarse-graining method {method!r}; choose: {supported}"
        )

    @classmethod
    def _build_backbone_sidechain_mapping(
        cls,
        atomgroup: AtomGroup,
        n_constraints: float = 0.0,
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

    @classmethod
    def _build_martini_mapping(
        cls,
        atomgroup: AtomGroup,
        n_constraints: float = 0.0,
    ) -> CoarseGrainMap:
        raise NotImplementedError(
            "Martini coarse-graining (method='martini') is not implemented yet."
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
        com = np.average(coords, axis=0, weights=weights)
        return np.asarray(com, dtype=np.float32)

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

        residue_bead_counts = np.bincount(
            mapping.resindices, minlength=n_residues
        )
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
        if (
            self._memory_universe is not None
            and self._memory_n_frames == n_frames
        ):
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
            "; pyFRESEAN coarse-grained topology",
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
            # Load topology from .top and coordinates from the trajectory only.
            # Passing the companion .gro as well makes MDAnalysis treat it as an
            # extra first frame (len = 1 + n_traj), duplicating frame 0.
            u = mda.Universe(
                output_top,
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
        timings: Optional[dict[str, float]] = None,
    ) -> mda.Universe:
        trajectory = self.atomgroup.universe.trajectory
        ag = self.atomgroup
        scratch = self.empty_cg_universe(n_frames=1)
        has_velocities = None

        with mda.Writer(output_traj, n_atoms=scratch.atoms.n_atoms) as writer:
            for frame_idx, frame in enumerate(frame_list):
                if timings is not None:
                    t_process = time.perf_counter()
                trajectory[frame]
                bead_pos = self.map_positions(ag.positions)
                if has_velocities is None:
                    has_velocities = trajectory.ts.has_velocities
                bead_vel = (
                    self.map_velocities(ag.velocities)
                    if has_velocities
                    else None
                )
                dims = trajectory.ts.dimensions
                box = (
                    None if dims is None else np.asarray(dims, dtype=np.float32)
                )
                if timings is not None:
                    timings[BENCH_T_FRAME_PROCESSING] += (
                        time.perf_counter() - t_process
                    )
                    t_write = time.perf_counter()
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
                if timings is not None:
                    timings[BENCH_T_WRITE_OUTPUTS] += (
                        time.perf_counter() - t_write
                    )

        if timings is not None:
            t_assemble = time.perf_counter()
        universe = self._load_cg_universe_from_disk(output_top, output_traj)
        if timings is not None:
            timings[BENCH_T_ASSEMBLE_UNIVERSE] += (
                time.perf_counter() - t_assemble
            )
        return universe

    def _write_cg_outputs(
        self,
        output_cg_map: Optional[str] = None,
    ) -> None:
        if output_cg_map is not None:
            self.save_mapping(output_cg_map)

    def build_universe(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        frames: Optional[Iterable[int]] = None,
        output_cg_trajectory: Optional[str] = None,
        output_cg_topology: Optional[str] = None,
        in_memory: bool = True,
        output_cg_map: Optional[str] = None,
        benchmark: bool = False,
        timings: Optional[dict[str, float]] = None,
    ) -> mda.Universe:
        """Build a CG universe from the source atom group already on this instance.

        Prefer :meth:`cg_universe` when starting from all-atom file paths.

        Parameters
        ----------
        benchmark
            If ``True``, record wall times on :attr:`CoarseGrain.benchmark`.
            Keys: ``bench_t_mapping``, ``bench_t_frame_processing``,
            ``bench_t_write_outputs``, ``bench_t_assemble_universe``, and
            ``bench_t_cg_total``.
        timings
            Optional timing dict populated when benchmarking. When omitted and
            ``benchmark=True``, a new dict is created on this instance.
        """
        if benchmark and timings is None:
            timings = new_cg_benchmark_timings()
        self._ensure_mapping(timings=timings)
        trajectory = self.atomgroup.universe.trajectory
        ag = self.atomgroup
        frame_list = self._resolve_frame_list(
            trajectory, start, stop, step, frames
        )

        if not frame_list:
            if timings is not None:
                t_write = time.perf_counter()
            self._write_cg_outputs(output_cg_map)
            if timings is not None:
                timings[BENCH_T_WRITE_OUTPUTS] += time.perf_counter() - t_write
                self.benchmark = finalize_cg_benchmark(timings)
            return self.empty_cg_universe(n_frames=0)

        if not in_memory:
            if output_cg_trajectory is None or output_cg_topology is None:
                raise ValueError(
                    "in_memory=False requires both output_cg_trajectory and "
                    "output_cg_topology"
                )
            universe = self._cg_universe_to_disk(
                frame_list,
                output_cg_topology,
                output_cg_trajectory,
                timings=timings,
            )
            if timings is not None:
                t_write = time.perf_counter()
            self._write_cg_outputs(output_cg_map)
            if timings is not None:
                timings[BENCH_T_WRITE_OUTPUTS] += time.perf_counter() - t_write
                self.benchmark = finalize_cg_benchmark(timings)
            return universe

        positions = []
        velocities = []
        dimensions_list: list[Optional[np.ndarray]] = []
        has_velocities = None

        for frame in frame_list:
            if timings is not None:
                t_process = time.perf_counter()
            trajectory[frame]
            positions.append(self.map_positions(ag.positions))
            if output_cg_trajectory is not None:
                dims = trajectory.ts.dimensions
                dimensions_list.append(
                    None if dims is None else np.asarray(dims, dtype=np.float32)
                )
            if has_velocities is None:
                has_velocities = trajectory.ts.has_velocities
            if has_velocities:
                velocities.append(self.map_velocities(ag.velocities))
            if timings is not None:
                timings[BENCH_T_FRAME_PROCESSING] += (
                    time.perf_counter() - t_process
                )

        if timings is not None:
            t_assemble = time.perf_counter()

        n_frames = len(positions)
        u = self.empty_cg_universe(n_frames=n_frames)
        pos_array = np.asarray(positions, dtype=np.float32)
        for frame_idx, ts in enumerate(u.trajectory):
            ts.positions = pos_array[frame_idx]
            if has_velocities:
                ts.velocities = np.asarray(
                    velocities[frame_idx], dtype=np.float32
                )

        if timings is not None:
            timings[BENCH_T_ASSEMBLE_UNIVERSE] += (
                time.perf_counter() - t_assemble
            )
            t_write = time.perf_counter()

        if output_cg_topology is not None:
            self._warn_topology_format(output_cg_topology)
            self.write_topology(u, output_cg_topology)
        if output_cg_trajectory is not None:
            self.write_trajectory(
                u, output_cg_trajectory, dimensions=dimensions_list
            )

        self._write_cg_outputs(output_cg_map)
        if timings is not None:
            timings[BENCH_T_WRITE_OUTPUTS] += time.perf_counter() - t_write
            self.benchmark = finalize_cg_benchmark(timings)
        return u

    def backmap_modes(self, cg_vectors: np.ndarray) -> np.ndarray:
        """Expand a coarse-grained mass-weighted mode onto atoms.

        ``cg_vectors`` use the same scaling as FRESEAN eigenvectors. Bead
        component ``u_b`` is ``sqrt(M_b)`` times that bead's Cartesian
        direction. Atom ``i`` receives ``u_i = u_b * sqrt(m_i / M_b)``,
        which keeps that scaling and gives every atom in the bead the
        same Cartesian direction.
        """
        mapping = self._ensure_mapping()
        if cg_vectors.ndim == 1:
            cg_vectors = cg_vectors.reshape(mapping.n_beads, 3)
        if cg_vectors.shape != (mapping.n_beads, 3):
            raise ValueError(
                f"expected cg_vectors shape ({mapping.n_beads}, 3), "
                f"got {cg_vectors.shape}"
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
            weights = np.sqrt(masses[indices] / bead_mass)[:, np.newaxis]
            aa_vectors[indices] = cg_vectors[bead_idx] * weights
        return aa_vectors

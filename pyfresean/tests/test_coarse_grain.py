import numpy as np
import pytest
import MDAnalysis as mda

from pyfresean.coarsegrain import CoarseGrain
from pyfresean.exceptions import MissingBeadMassWarning, TopologyFormatWarning


def _make_protein_universe(n_frames: int = 2):
    """Two residues: ALA (backbone + sidechain) and GLY (backbone only)."""
    names = ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O"]
    masses = [14.0, 12.0, 12.0, 16.0, 12.0, 14.0, 12.0, 12.0, 16.0]

    u = mda.Universe.empty(
        n_atoms=len(names),
        n_residues=2,
        n_segments=1,
        atom_resindex=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
        residue_segindex=np.array([0, 0], dtype=np.int32),
        trajectory=n_frames > 0,
        velocities=n_frames > 0,
    )
    u.add_TopologyAttr("names")
    u.add_TopologyAttr("resnames")
    u.add_TopologyAttr("resids")
    u.add_TopologyAttr("masses")
    u.atoms.names = names
    u.atoms.masses = masses
    u.residues.resnames = ["ALA", "GLY"]
    u.residues.resids = [1, 2]

    if n_frames > 0:
        positions = np.array(
            [[[i, 0.0, 0.0] for i in range(len(names))]] * n_frames,
            dtype=np.float32,
        )
        for frame in range(n_frames):
            positions[frame, :, 0] += frame * 10.0
        velocities = positions + 1.0
        from MDAnalysis.coordinates.memory import MemoryReader

        u.trajectory = MemoryReader(positions, velocities=velocities)
    return u


def test_mapping_built_on_first_access():
    u = _make_protein_universe()
    cg = CoarseGrain(u.atoms)
    assert cg._mapping is None
    assert cg.mapping.n_beads == 3
    assert cg._mapping is not None


def test_from_atomgroup_builds_mapping_eagerly():
    u = _make_protein_universe()
    cg = CoarseGrain.from_atomgroup(u.atoms)
    assert cg._mapping is not None
    assert cg.mapping.n_beads == 3


def test_build_canonical_mapping_bead_counts():
    u = _make_protein_universe()
    mapping = CoarseGrain.from_atomgroup(u.atoms).mapping
    assert mapping.n_beads == 3
    assert mapping.bead_types == ["BACK", "SIDE", "BACK"]
    assert mapping.bead_masses[0] == pytest.approx(54.0)
    assert mapping.bead_masses[1] == pytest.approx(12.0)


def test_map_positions_matches_hand_com():
    u = _make_protein_universe()
    u.atoms.positions = np.arange(u.atoms.n_atoms * 3, dtype=np.float32).reshape(-1, 3)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    bead_pos = cg.map_positions(u.atoms.positions)

    back_atoms = u.atoms[[0, 1, 2, 3]]
    expected_back = np.average(back_atoms.positions, axis=0, weights=back_atoms.masses)
    np.testing.assert_allclose(bead_pos[0], expected_back)


def test_backmap_guesses_missing_masses_from_atom_names():
    u = _make_protein_universe()
    u.atoms.masses = np.zeros(u.atoms.n_atoms, dtype=np.float32)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    masses = cg._atom_masses()
    assert masses[0] == pytest.approx(14.007)  # N
    assert masses[1] == pytest.approx(12.011)  # CA

    cg_vectors = np.ones((cg.mapping.n_beads, 3), dtype=np.float64)
    aa_vectors = cg.backmap_modes(cg_vectors)
    assert aa_vectors.shape == (u.atoms.n_atoms, 3)
    assert np.all(np.isfinite(aa_vectors))


def test_scale_plumed_directions_formula():
    u = _make_protein_universe()
    cg = CoarseGrain.from_atomgroup(u.atoms)
    masses = cg._atom_masses()
    aa_vectors = np.arange(u.atoms.n_atoms * 3, dtype=np.float64).reshape(-1, 3)
    scaled = cg.scale_plumed_directions(aa_vectors, scale=100.0)
    expected = 100.0 * aa_vectors * np.sqrt(masses)[:, np.newaxis]
    np.testing.assert_allclose(scaled, expected)


def test_write_plumed_mode_input(tmp_path):
    u = _make_protein_universe()
    u.atoms.positions = np.arange(u.atoms.n_atoms * 3, dtype=np.float32).reshape(-1, 3)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    mode_a = np.ones((u.atoms.n_atoms, 3), dtype=np.float64)
    mode_b = 2.0 * mode_a
    out_path = tmp_path / "plumed-mode-input.pdb"

    cg.write_plumed_mode_input([mode_a, mode_b], str(out_path))

    text = out_path.read_text(encoding="utf-8")
    assert text.count("REMARK TYPE=DIRECTION\n") == 2
    assert text.count("REMARK TYPE=OPTIMAL\n") == 1
    assert text.count("END\n") == 3
    directions = cg.scale_plumed_directions(mode_a)
    first_direction_line = (
        f"{directions[0,0]:8.3f}{directions[0,1]:8.3f}{directions[0,2]:8.3f}"
    )
    assert first_direction_line in text


def test_write_plumed_direction_pdb(tmp_path):
    u = _make_protein_universe()
    cg = CoarseGrain.from_atomgroup(u.atoms)
    aa_vectors = np.ones((u.atoms.n_atoms, 3), dtype=np.float64)
    out_path = tmp_path / "evec_1_aa_scaled.pdb"
    cg.write_plumed_direction_pdb(aa_vectors, str(out_path))
    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("REMARK TYPE=DIRECTION\n")
    assert text.endswith("END\n")


def test_backmap_mass_weighted_formula():
    u = _make_protein_universe()
    cg = CoarseGrain.from_atomgroup(u.atoms)
    masses = u.atoms.masses.astype(np.float64)
    cg_vectors = np.arange(cg.mapping.n_beads * 3, dtype=np.float64).reshape(-1, 3)
    aa_vectors = cg.backmap_modes(cg_vectors)

    for bead_idx, indices in enumerate(cg.mapping.atom_indices):
        bead_mass = cg.mapping.bead_masses[bead_idx]
        expected = cg_vectors[bead_idx] * np.sqrt(masses[indices] / bead_mass)[:, np.newaxis]
        np.testing.assert_allclose(aa_vectors[indices], expected)

        physical = aa_vectors[indices] / np.sqrt(masses[indices])[:, np.newaxis]
        np.testing.assert_allclose(
            physical,
            np.broadcast_to(
                cg_vectors[bead_idx] / np.sqrt(bead_mass),
                physical.shape,
            ),
        )


def test_cg_universe_from_files(tmp_path):
    u = _make_protein_universe(n_frames=3)
    aa_top = tmp_path / "aa.gro"
    aa_traj = tmp_path / "aa.trr"
    u.atoms.write(str(aa_top))
    with mda.Writer(str(aa_traj), n_atoms=u.atoms.n_atoms) as writer:
        for _ts in u.trajectory:
            writer.write(u.atoms)

    cg_top = tmp_path / "cg.top"
    cg_traj = tmp_path / "cg.trr"
    cg, u_cg = CoarseGrain.cg_universe(
        aa_top,
        aa_traj,
        select="all",
        output_top=str(cg_top),
        output_traj=str(cg_traj),
    )

    assert cg.mapping.n_beads == 3
    assert len(u_cg.trajectory) == 3
    np.testing.assert_allclose(u_cg.atoms.masses, cg.mapping.bead_masses)


def test_cg_universe_shapes():
    u = _make_protein_universe(n_frames=3)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    u_cg = cg.build_universe()

    assert len(u_cg.trajectory) == 3
    assert u_cg.atoms.n_atoms == 3
    assert u_cg.trajectory.ts.has_velocities

    u.trajectory[0]
    cg_bead_pos = cg.map_positions(u.atoms.positions)
    np.testing.assert_allclose(u_cg.trajectory[0].positions, cg_bead_pos)


def test_cg_universe_writes_standard_files(tmp_path):
    u = _make_protein_universe(n_frames=2)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    top_path = tmp_path / "cg.pdb"
    traj_path = tmp_path / "cg.trr"

    u_cg = cg.build_universe(output_top=str(top_path), output_traj=str(traj_path))

    assert top_path.is_file()
    assert traj_path.is_file()
    assert u_cg.atoms.n_atoms == 3

    u_top = mda.Universe(str(top_path))
    u_traj = mda.Universe(str(top_path), str(traj_path))
    assert u_top.atoms.n_atoms == 3
    assert len(u_traj.trajectory) == 2
    assert u_traj.trajectory.ts.has_velocities


def test_cg_universe_writes_standard_files(tmp_path):
    u = _make_protein_universe(n_frames=2)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    top_path = tmp_path / "cg.pdb"
    traj_path = tmp_path / "cg.trr"

    u_cg = cg.build_universe(output_top=str(top_path), output_traj=str(traj_path))

    assert top_path.is_file()
    assert traj_path.is_file()
    assert u_cg.atoms.n_atoms == 3

    u_top = mda.Universe(str(top_path))
    u_traj = mda.Universe(str(top_path), str(traj_path))
    assert u_top.atoms.n_atoms == 3
    assert len(u_traj.trajectory) == 2
    assert u_traj.trajectory.ts.has_velocities


def test_cg_universe_writes_top_file(tmp_path):
    u = _make_protein_universe(n_frames=2)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    top_path = tmp_path / "cg.top"
    gro_path = tmp_path / "cg.gro"
    traj_path = tmp_path / "cg.trr"

    u_cg = cg.build_universe(
        output_top=str(top_path),
        output_traj=str(traj_path),
        in_memory=False,
    )

    assert top_path.is_file()
    assert gro_path.is_file()
    np.testing.assert_allclose(u_cg.atoms.masses, cg.mapping.bead_masses)

    u_reload = mda.Universe(
        str(top_path),
        str(gro_path),
        str(traj_path),
        topology_format="ITP",
    )
    cg._restore_bead_masses(u_reload)
    np.testing.assert_allclose(u_reload.atoms.masses, cg.mapping.bead_masses)


def test_pdb_reload_warns_when_masses_restored(tmp_path):
    u = _make_protein_universe(n_frames=2)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    pdb_path = tmp_path / "cg.pdb"
    traj_path = tmp_path / "cg.trr"

    with pytest.warns((TopologyFormatWarning, MissingBeadMassWarning)):
        cg.build_universe(
            output_top=str(pdb_path),
            output_traj=str(traj_path),
            in_memory=False,
        )

    u_reload = mda.Universe(str(pdb_path), str(traj_path))
    assert not cg._bead_masses_trusted(u_reload)
    with pytest.warns(MissingBeadMassWarning):
        cg._restore_bead_masses(u_reload)
    np.testing.assert_allclose(u_reload.atoms.masses, cg.mapping.bead_masses)


def test_cg_universe_file_backed_without_memory_reader(tmp_path):
    from MDAnalysis.coordinates.memory import MemoryReader

    u = _make_protein_universe(n_frames=3)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    top_path = tmp_path / "cg.pdb"
    traj_path = tmp_path / "cg.trr"

    with pytest.warns((TopologyFormatWarning, MissingBeadMassWarning)):
        u_cg = cg.build_universe(
            output_top=str(top_path),
            output_traj=str(traj_path),
            in_memory=False,
        )

    assert not isinstance(u_cg.trajectory, MemoryReader)
    assert len(u_cg.trajectory) == 3
    assert u_cg.trajectory.ts.has_velocities
    np.testing.assert_allclose(u_cg.atoms.masses, cg.mapping.bead_masses)

    u.trajectory[0]
    expected = cg.map_positions(u.atoms.positions)
    np.testing.assert_allclose(u_cg.trajectory[0].positions, expected, rtol=1e-4)


def test_cg_universe_in_memory_false_requires_output_paths():
    u = _make_protein_universe(n_frames=2)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    with pytest.raises(ValueError, match="output_traj and output_top"):
        cg.build_universe(in_memory=False)


def test_empty_cg_universe_reuses_cached_shells():
    u = _make_protein_universe()
    cg = CoarseGrain.from_atomgroup(u.atoms)

    top_a = cg.empty_cg_universe(n_frames=0)
    top_b = cg.empty_cg_universe(n_frames=0)
    assert top_a is top_b

    scratch_a = cg.empty_cg_universe(n_frames=1)
    scratch_b = cg.empty_cg_universe(n_frames=1)
    assert scratch_a is scratch_b

    mem_a = cg.empty_cg_universe(n_frames=4)
    mem_b = cg.empty_cg_universe(n_frames=4)
    assert mem_a is mem_b

    mem_c = cg.empty_cg_universe(n_frames=2)
    assert mem_c is not mem_a


def test_fresean_runs_on_cg_trajectory():
    from pyfresean import FRESEAN

    u = _make_protein_universe(n_frames=4)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    u_cg = cg.build_universe()

    analysis = FRESEAN(
        u_cg,
        select="all",
        n_constraints=cg.mapping.n_constraints,
        n_corr=4,
        dt=0.02,
        sigma=10.0,
    )
    analysis.run()
    assert analysis.results.vdos_total.shape[0] == 4

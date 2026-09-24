import numpy as np
import pytest
import MDAnalysis as mda

from pyfresean.coarsegrain import (
    CoarseGrain,
    CENTERED_MODES,
    SUPPORTED_CG_METHODS,
)
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


def test_cg_universe_requires_input():
    with pytest.raises(TypeError):
        CoarseGrain.cg_universe()  # pylint: disable=no-value-for-parameter


def test_resolve_universe_accepts_universe_path_and_tuple(tmp_path):
    u = _make_protein_universe(n_frames=2)
    top = tmp_path / "aa.gro"
    traj = tmp_path / "aa.trr"
    u.atoms.write(str(top))
    with mda.Writer(str(traj), n_atoms=u.atoms.n_atoms) as writer:
        for _ts in u.trajectory:
            writer.write(u.atoms)

    assert CoarseGrain._resolve_universe(u) is u
    assert (
        CoarseGrain._resolve_universe(str(top)).atoms.n_atoms == u.atoms.n_atoms
    )
    assert len(CoarseGrain._resolve_universe((top, traj)).trajectory) == 2


def test_cg_universe_from_aa_universe():
    u = _make_protein_universe(n_frames=2)
    cg, u_cg = CoarseGrain.cg_universe(aa=u, select="all")
    assert cg.mapping.n_beads == 3
    assert len(u_cg.trajectory) == 2


def test_mapping_stores_orientational_reference():
    u = _make_protein_universe()
    mapping = CoarseGrain.from_atomgroup(u.atoms).mapping
    assert len(mapping.reference_centered_coords) == mapping.n_beads


def test_reconstruct_aa_universe_roundtrip_velocities(tmp_path):
    u = _make_protein_universe(n_frames=3)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    rot_path = tmp_path / "aa_rotations.npz"
    u_cg = cg.build_universe(output_aa_rotations=str(rot_path))
    u_aa = cg.reconstruct_aa_universe(u_cg=u_cg, centered=rot_path)

    for frame in range(3):
        u.trajectory[frame]
        u_aa.trajectory[frame]
        np.testing.assert_allclose(
            u_aa.atoms.positions,
            u.atoms.positions,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            u_aa.atoms.velocities,
            u.atoms.velocities,
            atol=1e-4,
        )


def test_reconstruct_aa_universe_roundtrip(tmp_path):
    u = _make_protein_universe(n_frames=3)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    rot_path = tmp_path / "aa_rotations.npz"
    u_cg = cg.build_universe(output_aa_rotations=str(rot_path))
    u_aa = cg.reconstruct_aa_universe(u_cg=u_cg, centered=rot_path)

    assert len(u_aa.trajectory) == 3
    assert u_aa.atoms.n_atoms == u.atoms.n_atoms
    for frame in range(3):
        u.trajectory[frame]
        u_aa.trajectory[frame]
        np.testing.assert_allclose(
            u_aa.atoms.positions,
            u.atoms.positions,
            atol=1e-4,
        )


def test_aa_universe_from_cg_files(tmp_path):
    u = _make_protein_universe(n_frames=2)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    cg_top = tmp_path / "cg.pdb"
    cg_traj = tmp_path / "cg.trr"
    u_cg = cg.build_universe(
        output_cg_topology=str(cg_top), output_cg_trajectory=str(cg_traj)
    )

    u_aa = CoarseGrain.aa_universe_from_cg(
        mapping=cg,
        u_cg=(cg_top, cg_traj),
    )
    assert len(u_aa.trajectory) == 2

    u.trajectory[0]
    u_aa.trajectory[0]
    np.testing.assert_allclose(
        u_aa.atoms.positions,
        u.atoms.positions,
        atol=1e-3,
    )


def test_mapping_built_on_first_access():
    u = _make_protein_universe()
    cg = CoarseGrain(u.atoms)
    assert cg._mapping is None
    assert cg.method == "backbone-sidechain"
    assert cg.mapping.n_beads == 3
    assert cg._mapping is not None


def test_backbone_sidechain_method_alias():
    u = _make_protein_universe()
    cg = CoarseGrain.from_atomgroup(u.atoms, method="backbone_sidechain")
    assert cg.method == "backbone-sidechain"
    assert cg.mapping.n_beads == 3


def test_invalid_cg_method_raises():
    u = _make_protein_universe()
    with pytest.raises(ValueError, match="unsupported coarse-graining method"):
        CoarseGrain.from_atomgroup(u.atoms, method="united-atom")


def test_martini_method_not_implemented_yet():
    u = _make_protein_universe()
    with pytest.raises(NotImplementedError, match="Martini coarse-graining"):
        CoarseGrain.from_atomgroup(u.atoms, method="martini")


def test_supported_cg_methods_constant():
    assert SUPPORTED_CG_METHODS == ["backbone-sidechain", "martini"]
    assert CoarseGrain.CG_METHODS == SUPPORTED_CG_METHODS


def test_centered_modes_constant():
    assert CENTERED_MODES == ["track", "ref"]


def test_mapping_stores_reference_centered_coords():
    u = _make_protein_universe()
    mapping = CoarseGrain.from_atomgroup(u.atoms).mapping
    assert len(mapping.reference_centered_coords) == mapping.n_beads


def test_reference_frame_for_mapping(tmp_path):
    u = _make_protein_universe(n_frames=3)
    cg = CoarseGrain.from_atomgroup(u.atoms, reference=2)
    assert cg.reference_frame == 2
    assert cg.mapping.reference_frame == 2


def test_centered_mode_ref_skips_tracked_vectors():
    u = _make_protein_universe(n_frames=3)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    cg.build_universe(centered_mode="ref")
    assert cg.centered_mode == "ref"
    assert cg.mapping.reference_centered_coords


def test_save_and_load_cg_map(tmp_path):
    u = _make_protein_universe()
    cg = CoarseGrain.from_atomgroup(u.atoms, reference=0)
    map_path = tmp_path / "cg_map.npz"
    cg.save_mapping(map_path)

    loaded_mapping, method = CoarseGrain.load_mapping(map_path)
    assert method == "backbone-sidechain"
    assert loaded_mapping.n_beads == cg.mapping.n_beads
    assert len(loaded_mapping.reference_centered_coords) == cg.mapping.n_beads

    cg_reload = CoarseGrain.from_cg_map(u.atoms, map_path)
    assert cg_reload.mapping.n_beads == cg.mapping.n_beads


def test_save_and_load_centered_deltas(tmp_path):
    u = _make_protein_universe(n_frames=3)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    rot_path = tmp_path / "aa_rotations.npz"
    cg.build_universe(centered_mode="track", output_aa_rotations=str(rot_path))

    centered, _, mode = CoarseGrain.load_centered_deltas(rot_path)
    assert mode == "track"
    assert centered is not None
    assert centered.shape == (3, u.atoms.n_atoms, 3)


def test_cg_universe_writes_map_and_rotations(tmp_path):
    u = _make_protein_universe(n_frames=2)
    map_path = tmp_path / "cg_map.npz"
    rot_path = tmp_path / "aa_rotations.npz"
    cg, u_cg = CoarseGrain.cg_universe(
        aa=u,
        centered_mode="track",
        output_cg_map=str(map_path),
        output_aa_rotations=str(rot_path),
    )
    assert map_path.is_file()
    assert rot_path.is_file()
    assert len(u_cg.trajectory) == 2

    cg_reload = CoarseGrain.from_cg_map(u.atoms, map_path)
    u_aa = cg_reload.reconstruct_aa_universe(u_cg=u_cg, centered=rot_path)
    for frame in range(2):
        u.trajectory[frame]
        u_aa.trajectory[frame]
        np.testing.assert_allclose(
            u_aa.atoms.positions, u.atoms.positions, atol=1e-4
        )


def test_reconstruct_aa_universe_from_file_paths(tmp_path):
    u = _make_protein_universe(n_frames=2)
    map_path = tmp_path / "cg_map.npz"
    rot_path = tmp_path / "aa_rotations.npz"
    cg_top = tmp_path / "cg.pdb"
    cg_traj = tmp_path / "cg.trr"
    cg, u_cg = CoarseGrain.cg_universe(
        aa=u,
        output_cg_map=str(map_path),
        output_aa_rotations=str(rot_path),
        output_cg_topology=str(cg_top),
        output_cg_trajectory=str(cg_traj),
    )

    u_aa = cg.reconstruct_aa_universe(
        u_cg=(cg_top, cg_traj),
        centered=rot_path,
    )
    u.trajectory[0]
    u_aa.trajectory[0]
    np.testing.assert_allclose(
        u_aa.atoms.positions, u.atoms.positions, atol=1e-3
    )


def test_track_backmap_from_top_trr_matches_source(tmp_path):
    u = _make_protein_universe(n_frames=5)
    map_path = tmp_path / "cg_map.npz"
    rot_path = tmp_path / "aa_rotations.npz"
    cg_top = tmp_path / "cg.top"
    cg_traj = tmp_path / "cg.trr"
    cg, _ = CoarseGrain.cg_universe(
        aa=u,
        output_cg_map=str(map_path),
        output_aa_rotations=str(rot_path),
        output_cg_topology=str(cg_top),
        output_cg_trajectory=str(cg_traj),
    )

    u_cg_raw = mda.Universe(str(cg_top), str(cg_traj), topology_format="ITP")
    cg._restore_bead_masses(u_cg_raw)
    u_aa = cg.reconstruct_aa_universe(u_cg=u_cg_raw, centered=rot_path)
    for frame in range(len(u.trajectory)):
        u.trajectory[frame]
        u_aa.trajectory[frame]
        np.testing.assert_allclose(
            u_aa.atoms.positions, u.atoms.positions, atol=1e-3
        )


def test_aa_universe_from_cg_with_atomgroup_and_map(tmp_path):
    u = _make_protein_universe(n_frames=2)
    map_path = tmp_path / "cg_map.npz"
    rot_path = tmp_path / "aa_rotations.npz"
    cg_top = tmp_path / "cg.pdb"
    cg_traj = tmp_path / "cg.trr"
    CoarseGrain.cg_universe(
        aa=u,
        output_cg_map=str(map_path),
        output_aa_rotations=str(rot_path),
        output_cg_topology=str(cg_top),
        output_cg_trajectory=str(cg_traj),
    )

    u_aa = CoarseGrain.aa_universe_from_cg(
        u_cg=(cg_top, cg_traj),
        mapping=(u.atoms, map_path),
        centered=rot_path,
    )
    u.trajectory[0]
    u_aa.trajectory[0]
    np.testing.assert_allclose(
        u_aa.atoms.positions, u.atoms.positions, atol=1e-3
    )


def test_aa_universe_from_cg_with_cg_and_u_cg(tmp_path):
    u = _make_protein_universe(n_frames=2)
    rot_path = tmp_path / "aa_rotations.npz"
    cg, u_cg = CoarseGrain.cg_universe(
        aa=u,
        output_aa_rotations=str(rot_path),
    )
    u_aa = CoarseGrain.aa_universe_from_cg(
        mapping=cg,
        u_cg=u_cg,
        centered=rot_path,
    )
    u.trajectory[0]
    u_aa.trajectory[0]
    np.testing.assert_allclose(
        u_aa.atoms.positions, u.atoms.positions, atol=1e-4
    )


def test_track_mode_writes_rotation_file_not_instance(tmp_path):
    u = _make_protein_universe(n_frames=3)
    rot_path = tmp_path / "aa_rotations.npz"
    cg = CoarseGrain.from_atomgroup(u.atoms)
    cg.build_universe(centered_mode="track", output_aa_rotations=str(rot_path))
    assert rot_path.is_file()
    centered, _, mode = CoarseGrain.load_centered_deltas(rot_path)
    assert mode == "track"
    assert centered.shape == (3, u.atoms.n_atoms, 3)

    cg_ref = CoarseGrain.from_atomgroup(u.atoms)
    cg_ref.build_universe(centered_mode="ref")
    assert cg_ref.mapping.reference_centered_coords


def test_aa_universe_from_cg_with_saved_rotations(tmp_path):
    u = _make_protein_universe(n_frames=2)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    cg_top = tmp_path / "cg.pdb"
    cg_traj = tmp_path / "cg.trr"
    rot_path = tmp_path / "aa_rotations.npz"
    u_cg = cg.build_universe(
        output_cg_topology=str(cg_top),
        output_cg_trajectory=str(cg_traj),
        output_aa_rotations=str(rot_path),
    )

    u_aa = CoarseGrain.aa_universe_from_cg(
        mapping=cg,
        u_cg=(cg_top, cg_traj),
        centered=rot_path,
    )
    u.trajectory[0]
    u_aa.trajectory[0]
    np.testing.assert_allclose(
        u_aa.atoms.positions, u.atoms.positions, atol=1e-3
    )


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


def test_backmap_positions_exact_with_tracked_centered():
    u = _make_protein_universe(n_frames=3)
    u.atoms.positions = np.arange(
        u.atoms.n_atoms * 3, dtype=np.float32
    ).reshape(-1, 3)
    for frame in range(3):
        u.trajectory[frame]
        u.atoms.positions += frame * 0.17
    cg = CoarseGrain.from_atomgroup(u.atoms)
    for frame in range(3):
        u.trajectory[frame]
        centered = cg.compute_centered_coords(u.atoms.positions)
        rebuilt = cg.backmap_positions(
            cg.map_positions(u.atoms.positions), centered
        )
        np.testing.assert_allclose(rebuilt, u.atoms.positions, atol=1e-5)


def test_backmap_positions_uses_reference_when_centered_omitted():
    u = _make_protein_universe()
    cg = CoarseGrain.from_atomgroup(u.atoms)
    com = cg.map_positions(u.atoms.positions)
    rebuilt = cg.backmap_positions(com)
    np.testing.assert_allclose(rebuilt, u.atoms.positions, atol=1e-5)


def test_map_positions_matches_hand_com():
    u = _make_protein_universe()
    u.atoms.positions = np.arange(
        u.atoms.n_atoms * 3, dtype=np.float32
    ).reshape(-1, 3)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    bead_pos = cg.map_positions(u.atoms.positions)

    back_atoms = u.select_atoms("index 0 1 2 3")
    expected_back = np.average(
        back_atoms.positions, axis=0, weights=back_atoms.masses
    )
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
    u.atoms.positions = np.arange(
        u.atoms.n_atoms * 3, dtype=np.float32
    ).reshape(-1, 3)
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
    cg_vectors = np.arange(cg.mapping.n_beads * 3, dtype=np.float64).reshape(
        -1, 3
    )
    aa_vectors = cg.backmap_modes(cg_vectors)

    for bead_idx, indices in enumerate(cg.mapping.atom_indices):
        bead_mass = cg.mapping.bead_masses[bead_idx]
        expected = (
            cg_vectors[bead_idx]
            * np.sqrt(masses[indices] / bead_mass)[:, np.newaxis]
        )
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
        aa=(aa_top, aa_traj),
        select="all",
        output_cg_topology=str(cg_top),
        output_cg_trajectory=str(cg_traj),
    )

    assert cg.mapping.n_beads == 3
    assert len(u_cg.trajectory) == 3
    np.testing.assert_allclose(u_cg.atoms.masses, cg.mapping.bead_masses)


def test_build_universe_benchmark_records_phase_timings():
    u = _make_protein_universe(n_frames=3)
    cg, u_cg = CoarseGrain.cg_universe(u, benchmark=True)

    assert len(u_cg.trajectory) == 3
    from pyfresean.benchmark_keys import (
        BENCH_T_ASSEMBLE_UNIVERSE,
        BENCH_T_FRAME_PROCESSING,
        BENCH_T_MAPPING,
        BENCH_T_CG_TOTAL,
        BENCH_T_WRITE_OUTPUTS,
    )

    timings = cg.benchmark
    assert set(timings) == {
        BENCH_T_MAPPING,
        BENCH_T_FRAME_PROCESSING,
        BENCH_T_WRITE_OUTPUTS,
        BENCH_T_ASSEMBLE_UNIVERSE,
        BENCH_T_CG_TOTAL,
    }
    for key in timings:
        assert timings[key] >= 0.0
    assert timings[BENCH_T_MAPPING] > 0.0
    assert timings[BENCH_T_FRAME_PROCESSING] > 0.0
    assert timings[BENCH_T_CG_TOTAL] == pytest.approx(
        timings[BENCH_T_MAPPING]
        + timings[BENCH_T_FRAME_PROCESSING]
        + timings[BENCH_T_WRITE_OUTPUTS]
        + timings[BENCH_T_ASSEMBLE_UNIVERSE]
    )


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

    u_cg = cg.build_universe(
        output_cg_topology=str(top_path), output_cg_trajectory=str(traj_path)
    )

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
        output_cg_topology=str(top_path),
        output_cg_trajectory=str(traj_path),
        in_memory=False,
    )

    assert top_path.is_file()
    assert gro_path.is_file()
    np.testing.assert_allclose(u_cg.atoms.masses, cg.mapping.bead_masses)

    u_reload = mda.Universe(
        str(top_path),
        str(traj_path),
        topology_format="ITP",
    )
    cg._restore_bead_masses(u_reload)
    np.testing.assert_allclose(u_reload.atoms.masses, cg.mapping.bead_masses)
    assert len(u_reload.trajectory) == len(u_cg.trajectory)


def test_pdb_reload_warns_when_masses_restored(tmp_path):
    u = _make_protein_universe(n_frames=2)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    pdb_path = tmp_path / "cg.pdb"
    traj_path = tmp_path / "cg.trr"

    with pytest.warns((TopologyFormatWarning, MissingBeadMassWarning)):
        cg.build_universe(
            output_cg_topology=str(pdb_path),
            output_cg_trajectory=str(traj_path),
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
            output_cg_topology=str(top_path),
            output_cg_trajectory=str(traj_path),
            in_memory=False,
        )

    assert not isinstance(u_cg.trajectory, MemoryReader)
    assert len(u_cg.trajectory) == 3
    assert u_cg.trajectory.ts.has_velocities
    np.testing.assert_allclose(u_cg.atoms.masses, cg.mapping.bead_masses)

    u.trajectory[0]
    expected = cg.map_positions(u.atoms.positions)
    np.testing.assert_allclose(
        u_cg.trajectory[0].positions, expected, rtol=1e-4
    )


def test_cg_universe_in_memory_false_requires_output_paths():
    u = _make_protein_universe(n_frames=2)
    cg = CoarseGrain.from_atomgroup(u.atoms)
    with pytest.raises(
        ValueError, match="output_cg_trajectory and output_cg_topology"
    ):
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

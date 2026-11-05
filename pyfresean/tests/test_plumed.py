import warnings

import numpy as np
import pytest

from pyfresean.coarsegrain import CoarseGrain
from pyfresean.exceptions import PlumedAllAtomWarning
from pyfresean.postprocess.plumed import (
    plumed_directions,
    projection_standard_deviations,
    write_plumed_metad_input,
    write_plumed_mode_input,
    write_plumed_projection_input,
)
from pyfresean.tests.test_coarse_grain import _make_protein_universe


def test_plumed_directions_multiplies_by_sqrt_mass():
    masses = np.array([4.0, 9.0])
    mode = np.ones((2, 3))
    got = plumed_directions(mode, masses, scale=100.0)
    expected = 100.0 * mode * np.sqrt(masses)[:, np.newaxis]
    np.testing.assert_allclose(got, expected)


def test_plumed_directions_default_scale_is_100():
    got = plumed_directions(np.array([[1.0, 0.0, 0.0]]), np.array([4.0]))
    np.testing.assert_allclose(got, [[200.0, 0.0, 0.0]])


def test_plumed_directions_rejects_bad_input():
    with pytest.raises(ValueError, match="positive"):
        plumed_directions(np.ones((2, 3)), np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="length"):
        plumed_directions(np.ones(5), np.array([1.0, 4.0]))


def test_plumed_directions_warns_for_bead_length():
    bead_masses = np.array([14.0, 36.0])
    mode = np.ones((2, 3))
    with pytest.warns(PlumedAllAtomWarning, match="all-atom"):
        got = plumed_directions(mode, bead_masses, n_atoms=9)
    expected = 100.0 * mode * np.sqrt(bead_masses)[:, np.newaxis]
    np.testing.assert_allclose(got, expected)


def test_write_plumed_mode_input(tmp_path):
    u = _make_protein_universe()
    u.atoms.positions = np.arange(
        u.atoms.n_atoms * 3, dtype=np.float32
    ).reshape(-1, 3)
    mode_a = np.ones((u.atoms.n_atoms, 3), dtype=np.float64)
    mode_b = 2.0 * mode_a
    out_path = tmp_path / "plumed-mode-input.pdb"

    write_plumed_mode_input(u.atoms, [mode_a, mode_b], str(out_path))

    text = out_path.read_text(encoding="utf-8")
    assert text.count("REMARK TYPE=DIRECTION\n") == 2
    assert text.count("REMARK TYPE=OPTIMAL\n") == 1
    assert text.count("END\n") == 3
    directions = plumed_directions(mode_a, u.atoms.masses)
    first_direction_line = (
        f"{directions[0, 0]:8.3f}{directions[0, 1]:8.3f}"
        f"{directions[0, 2]:8.3f}"
    )
    assert first_direction_line in text


def test_write_plumed_direction_only(tmp_path):
    u = _make_protein_universe()
    aa_vectors = np.ones((u.atoms.n_atoms, 3), dtype=np.float64)
    out_path = tmp_path / "evec_1_aa_scaled.pdb"
    with warnings.catch_warnings():
        warnings.simplefilter("error", PlumedAllAtomWarning)
        write_plumed_mode_input(
            u.atoms,
            [aa_vectors],
            str(out_path),
            include_reference=False,
        )
    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("REMARK TYPE=DIRECTION\n")
    assert "REMARK TYPE=OPTIMAL\n" not in text
    assert text.endswith("END\n")
    directions = plumed_directions(aa_vectors, u.atoms.masses)
    first_line = (
        f"{directions[0, 0]:8.3f}{directions[0, 1]:8.3f}"
        f"{directions[0, 2]:8.3f}"
    )
    assert first_line in text


def test_projection_standard_deviations():
    masses = np.array([4.0, 9.0])
    reference = np.zeros((2, 3))
    frames = np.zeros((3, 2, 3))
    frames[1, 0, 0] = 1.0
    frames[2, 0, 0] = 3.0
    frames[:, 1, 2] = [0.0, 1.0, 1.0]
    mode_a = np.zeros((2, 3))
    mode_a[0, 0] = 1.0
    mode_b = np.zeros((2, 3))
    mode_b[1, 2] = 1.0
    # PLUMED normalizes each DIRECTION, then dots it with dx in nm.
    expected = [
        np.std([0.0, 0.1, 0.3]),
        np.std([0.0, 0.1, 0.1]),
    ]

    got = projection_standard_deviations(
        frames, [mode_a, mode_b], masses, reference
    )
    np.testing.assert_allclose(got, expected)

    flat = projection_standard_deviations(
        frames.reshape(3, -1),
        np.stack([mode_a.reshape(-1), mode_b.reshape(-1)]),
        masses,
        reference.reshape(-1),
        scale=1.0,
    )
    np.testing.assert_allclose(flat, expected)

    with pytest.raises(ValueError, match="masses"):
        projection_standard_deviations(
            frames, [mode_a], np.array([4.0]), reference
        )
    with pytest.raises(ValueError, match="frame"):
        projection_standard_deviations(frames[:0], [mode_a], masses, reference)


def test_write_plumed_projection_input(tmp_path):
    out_path = tmp_path / "plumed-mode-projection.dat"
    write_plumed_projection_input(str(out_path), 2)
    text = out_path.read_text(encoding="utf-8")
    assert (
        "PCAVARS REFERENCE=plumed-mode-input.pdb TYPE=OPTIMAL LABEL=pca\n"
        in text
    )
    assert (
        "PRINT ARG=pca.eig-1,pca.eig-2 "
        "FILE=plumed-mode-projection.out STRIDE=1000\n"
    ) in text

    custom = tmp_path / "custom.dat"
    write_plumed_projection_input(
        str(custom),
        1,
        reference="modes.pdb",
        output="proj.out",
        stride=500,
    )
    custom_text = custom.read_text(encoding="utf-8")
    assert "PCAVARS REFERENCE=modes.pdb TYPE=OPTIMAL LABEL=pca\n" in custom_text
    assert "PRINT ARG=pca.eig-1 FILE=proj.out STRIDE=500\n" in custom_text

    with pytest.raises(ValueError, match="n_modes"):
        write_plumed_projection_input(str(tmp_path / "empty.dat"), 0)


def test_write_plumed_metad_input(tmp_path):
    out_path = tmp_path / "plumed-mode-metadyn.dat"
    write_plumed_metad_input(str(out_path), [0.001, 0.001])
    text = out_path.read_text(encoding="utf-8")
    assert "PCAVARS REFERENCE=plumed-mode-input.pdb TYPE=OPTIMAL LABEL=pca\n" in text
    assert "ARG=pca.eig-1,pca.eig-2\n" in text
    assert "PACE=500\n" in text
    assert "HEIGHT=0.1\n" in text
    assert "BIASFACTOR=10\n" in text
    assert "SIGMA=0.001,0.001\n" in text
    assert "FILE=plumed-mode-metadyn.hills\n" in text
    assert "TEMP=300\n" in text
    assert (
        "PRINT ARG=pca.eig-1,pca.eig-2,pca.residual,metad.bias "
        "FILE=plumed-mode-metadyn.out STRIDE=500\n"
    ) in text

    custom = tmp_path / "custom-metad.dat"
    write_plumed_metad_input(
        str(custom),
        [0.02],
        timestep=0.001,
        deposition=2.0,
        height=1.0,
        biasfactor=15.0,
        temperature=310.0,
        reference="modes.pdb",
        hills="hills.dat",
        output="metad.out",
    )
    custom_text = custom.read_text(encoding="utf-8")
    assert "PCAVARS REFERENCE=modes.pdb TYPE=OPTIMAL LABEL=pca\n" in custom_text
    assert "ARG=pca.eig-1\n" in custom_text
    assert "PACE=2000\n" in custom_text
    assert "HEIGHT=1\n" in custom_text
    assert "BIASFACTOR=15\n" in custom_text
    assert "SIGMA=0.02\n" in custom_text
    assert "TEMP=310\n" in custom_text
    assert "FILE=hills.dat\n" in custom_text
    assert "FILE=metad.out STRIDE=2000\n" in custom_text

    with pytest.raises(ValueError, match="sigmas"):
        write_plumed_metad_input(str(tmp_path / "empty.dat"), [])
    with pytest.raises(ValueError, match="deposition"):
        write_plumed_metad_input(
            str(tmp_path / "pace.dat"),
            [0.001],
            timestep=0.002,
            deposition=0.003,
        )


def test_write_plumed_mode_input_warns_for_bead_vectors(tmp_path):
    u = _make_protein_universe()
    cg = CoarseGrain.from_atomgroup(u.atoms)
    cg_vectors = np.ones((cg.mapping.n_beads, 3), dtype=np.float64)
    out_path = tmp_path / "evec_cg.pdb"
    with pytest.warns(PlumedAllAtomWarning, match="all-atom"):
        with pytest.raises(ValueError, match="shape"):
            write_plumed_mode_input(u.atoms, [cg_vectors], str(out_path))
    assert not out_path.exists()

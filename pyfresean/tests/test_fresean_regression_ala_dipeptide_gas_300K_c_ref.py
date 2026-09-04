"""c_ref tests: pyfresean vs stored FRESEAN COARSE (C) reference (ala dipeptide, gas, 300 K)."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from pyfresean.tests.c_ref.ala_dipeptide_gas_300K import (
    N_FRAMES_DEFAULT,
    compare_cg_trajectory_to_c_reference,
    compare_fresean_eigenvalues_to_c,
    compare_fresean_eigenvectors_to_c,
    compare_fresean_vdos_total_to_c,
    resolve_ala_dipeptide_gas_300K_paths,
    run_pyfresean_coarse_grain,
    run_pyfresean_fresean_cg,
)
from pyfresean.tests.c_ref.io import load_c_eigenvalues_dat
from pyfresean.tests.c_ref.paths import ala_dipeptide_gas_300K_reference_dir
from pyfresean.tests.c_ref.plotting import plot_eigenvalues_vs_c, plot_vdos_total_vs_c


@pytest.fixture(scope="module")
def ala_gas_300k_paths():
    paths = resolve_ala_dipeptide_gas_300K_paths()
    aa = paths.aa_dir
    if not (aa / "topol.tpr").exists() or not (aa / "traj.trr").exists():
        pytest.skip(
            f"MD-gas-300K not available under {paths.data_root} "
            "(see examples/input_data/README.md)"
        )
    return paths


@pytest.fixture(scope="module")
def c_cg_reference(ala_gas_300k_paths):
    traj = ala_gas_300k_paths.c_cg_traj
    if not traj.exists():
        pytest.skip(
            f"C CG trajectory missing: {traj}. "
            "Run: python devtools/scripts/generate_fresean_c_ref_ala_dipeptide_gas_300K.py"
        )
    return ala_gas_300k_paths


@pytest.fixture(scope="module")
def c_fresean_reference(ala_gas_300k_paths):
    ref_dir = ala_gas_300k_paths.c_reference_dir
    eval_dat = ref_dir / "eval_covar_cg.mmat.dat"
    evec_mmat = ref_dir / "evec_covar_cg.mmat"
    if not eval_dat.exists() or not evec_mmat.exists():
        pytest.skip(
            f"C FRESEAN reference missing under {ref_dir}. "
            "Run: python devtools/scripts/generate_fresean_c_ref_ala_dipeptide_gas_300K.py"
        )
    return ala_gas_300k_paths


@pytest.fixture(scope="module")
def pyfresean_cg(ala_gas_300k_paths):
    cg, u_cg = run_pyfresean_coarse_grain(ala_gas_300k_paths, n_frames=N_FRAMES_DEFAULT)
    return cg, u_cg


@pytest.fixture(scope="module")
def pyfresean_fresean(ala_gas_300k_paths, pyfresean_cg):
    cg, u_cg = pyfresean_cg
    analysis = run_pyfresean_fresean_cg(cg, u_cg, ala_gas_300k_paths)
    return analysis, ala_gas_300k_paths


@pytest.mark.slow
@pytest.mark.c_ref
class TestAlaDipeptideGas300KCoarseGrainVsC:
    def test_cg_trajectory_matches_c_reference(self, c_cg_reference, pyfresean_cg):
        paths = c_cg_reference
        _, u_cg = pyfresean_cg
        metrics = compare_cg_trajectory_to_c_reference(u_cg, paths.c_cg_traj)
        assert metrics["max_position_rmsd_nm"] < 0.05


@pytest.mark.slow
@pytest.mark.c_ref
class TestAlaDipeptideGas300KFreseanVsC:
    def test_fresean_eigenvalues_match_c_reference(
        self, c_fresean_reference, pyfresean_fresean
    ):
        analysis, paths = pyfresean_fresean
        metrics = compare_fresean_eigenvalues_to_c(analysis, paths.c_eval_dat)
        assert metrics["max_abs_diff"] < 1e-3

    def test_fresean_vdos_total_matches_c_reference(
        self, c_fresean_reference, pyfresean_fresean
    ):
        analysis, paths = pyfresean_fresean
        metrics = compare_fresean_vdos_total_to_c(analysis, paths.c_eval_dat)
        assert metrics["max_abs_diff"] < 1e-3

    def test_fresean_eigenvectors_match_c_reference(
        self, c_fresean_reference, pyfresean_fresean
    ):
        analysis, paths = pyfresean_fresean
        metrics = compare_fresean_eigenvectors_to_c(
            analysis,
            paths.c_evec_mmat,
            freq_indices=(0, 1),
            n_modes=12,
        )
        assert all(v >= 0.99 for v in metrics.values())

    def test_fresean_eigenvalues_plot_vs_c_reference(
        self,
        c_fresean_reference,
        pyfresean_fresean,
        request,
    ):
        if not request.config.getoption("--c-ref-plot"):
            pytest.skip("pass --c-ref-plot to write eigenvalue comparison plots")

        analysis, paths = pyfresean_fresean
        plot_dir_opt = request.config.getoption("--c-ref-plot-dir")
        plot_dir = (
            Path(plot_dir_opt)
            if plot_dir_opt
            else ala_dipeptide_gas_300K_reference_dir() / "plots"
        )
        output_png = plot_eigenvalues_vs_c(
            analysis,
            paths.c_eval_dat,
            plot_dir / "eigenvalues_pyfresean_vs_c.png",
            freq_indices=(0, 1),
            n_modes=12,
        )
        assert output_png.is_file()
        assert output_png.stat().st_size > 0

    def test_fresean_vdos_total_plot_vs_c_reference(
        self,
        c_fresean_reference,
        pyfresean_fresean,
        request,
    ):
        if not request.config.getoption("--c-ref-plot"):
            pytest.skip("pass --c-ref-plot to write VDoS comparison plots")

        analysis, paths = pyfresean_fresean
        plot_dir_opt = request.config.getoption("--c-ref-plot-dir")
        plot_dir = (
            Path(plot_dir_opt)
            if plot_dir_opt
            else ala_dipeptide_gas_300K_reference_dir() / "plots"
        )
        output_png = plot_vdos_total_vs_c(
            analysis,
            paths.c_eval_dat,
            plot_dir / "vdos_total_pyfresean_vs_c.png",
        )
        assert output_png.is_file()
        assert output_png.stat().st_size > 0


def test_ala_dipeptide_gas_300K_c_reference_dir():
    ref = ala_dipeptide_gas_300K_reference_dir()
    assert ref.name == "ala_dipeptide_gas_300K"
    assert ref.parent.name == "fresean_c_ref"


def test_load_c_eigenvalues_dat_roundtrip(tmp_path):
    eval_path = tmp_path / "eval.dat"
    data = np.array([[3.0, 2.0, 1.0], [0.5, 0.4, 0.3]])
    with eval_path.open("w") as handle:
        for row in data:
            handle.write(" ".join(f"{x:e}" for x in row) + "\n")
    loaded = load_c_eigenvalues_dat(eval_path)
    np.testing.assert_allclose(loaded, data)

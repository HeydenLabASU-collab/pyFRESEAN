"""c_ref tests: pyfresean vs stored FRESEAN COARSE (C) reference (HEWL, solution, 303 K)."""

from __future__ import annotations

import pytest
from pathlib import Path

from pyfresean.tests.c_ref.hewl_solution_303K import (
    N_FRAMES_DEFAULT,
    N_MODES_COMPARE,
    compare_cg_trajectory_to_c_reference,
    compare_fresean_eigenvalues_to_c,
    compare_fresean_eigenvectors_to_c,
    compare_fresean_vdos_total_to_c,
    resolve_hewl_solution_303K_paths,
    run_pyfresean_coarse_grain,
    run_pyfresean_fresean_cg,
)
from pyfresean.tests.c_ref.paths import hewl_solution_303K_reference_dir
from pyfresean.tests.c_ref.plotting import (
    plot_eigenvalues_vs_c,
    plot_vdos_total_vs_c,
)


@pytest.fixture(scope="module")
def hewl_303k_paths():
    paths = resolve_hewl_solution_303K_paths()
    if not paths.aa_topol.exists() or not paths.aa_traj.exists():
        pytest.skip(
            f"HEWL protein trajectory missing under {paths.data_dir}/output. "
            "Run GROMACS MD and postprocess-prot.sh first."
        )
    return paths


@pytest.fixture(scope="module")
def c_cg_reference(hewl_303k_paths):
    traj = hewl_303k_paths.c_cg_traj
    if not traj.exists():
        pytest.skip(
            f"C CG trajectory missing: {traj}. "
            "Run: python devtools/scripts/generate_fresean_c_ref_hewl_solution_303K.py"
        )
    return hewl_303k_paths


@pytest.fixture(scope="module")
def c_fresean_reference(hewl_303k_paths):
    ref_dir = hewl_303k_paths.c_reference_dir
    eval_dat = ref_dir / "eval_covar_cg.mmat.dat"
    evec_mmat = ref_dir / "evec_covar_cg.mmat"
    if not eval_dat.exists() or not evec_mmat.exists():
        pytest.skip(
            f"C FRESEAN reference missing under {ref_dir}. "
            "Run: python devtools/scripts/generate_fresean_c_ref_hewl_solution_303K.py"
        )
    return hewl_303k_paths


@pytest.fixture(scope="module")
def pyfresean_cg(hewl_303k_paths):
    return run_pyfresean_coarse_grain(
        hewl_303k_paths, n_frames=N_FRAMES_DEFAULT
    )


@pytest.fixture(scope="module")
def pyfresean_fresean(hewl_303k_paths, pyfresean_cg):
    cg, u_cg = pyfresean_cg
    analysis = run_pyfresean_fresean_cg(cg, u_cg)
    return analysis, hewl_303k_paths


@pytest.mark.slow
@pytest.mark.c_ref
class TestHewlSolution303KCoarseGrainVsC:
    def test_cg_trajectory_matches_c_reference(
        self, c_cg_reference, pyfresean_cg
    ):
        paths = c_cg_reference
        _, u_cg = pyfresean_cg
        metrics = compare_cg_trajectory_to_c_reference(u_cg, paths.c_cg_traj)
        assert metrics["max_position_rmsd_nm"] < 0.05


@pytest.mark.slow
@pytest.mark.c_ref
class TestHewlSolution303KFreseanVsC:
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
            n_modes=N_MODES_COMPARE,
        )
        assert all(v >= 0.99 for v in metrics.values())

    def test_fresean_eigenvalues_plot_vs_c_reference(
        self,
        c_fresean_reference,
        pyfresean_fresean,
        request,
    ):
        if not request.config.getoption("--c-ref-plot"):
            pytest.skip(
                "pass --c-ref-plot to write eigenvalue comparison plots"
            )

        analysis, paths = pyfresean_fresean
        plot_dir_opt = request.config.getoption("--c-ref-plot-dir")
        plot_dir = (
            Path(plot_dir_opt)
            if plot_dir_opt
            else hewl_solution_303K_reference_dir() / "plots"
        )
        output_png = plot_eigenvalues_vs_c(
            analysis,
            paths.c_eval_dat,
            plot_dir / "eigenvalues_pyfresean_vs_c.png",
            freq_indices=(0, 1),
            n_modes=N_MODES_COMPARE,
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
            else hewl_solution_303K_reference_dir() / "plots"
        )
        output_png = plot_vdos_total_vs_c(
            analysis,
            paths.c_eval_dat,
            plot_dir / "vdos_total_pyfresean_vs_c.png",
        )
        assert output_png.is_file()
        assert output_png.stat().st_size > 0


def test_hewl_solution_303K_c_reference_dir():
    ref = hewl_solution_303K_reference_dir()
    assert ref.name == "hewl_solution_303K"
    assert ref.parent.name == "fresean_c_ref"

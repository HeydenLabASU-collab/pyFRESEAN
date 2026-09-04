"""Regression tests for alanine dipeptide (gas, 50 K) FRESEAN vs tutorial reference."""

from __future__ import annotations

import numpy as np
import pytest
import MDAnalysis as mda

from pyfresean import Align, FRESEAN
from pyfresean.postprocess import low_frequency_peaks
from pyfresean.tests.pytutorial_ref.ala_dipeptide_gas_50K import ensure_gas_50k_data
from pyfresean.tests.pytutorial_ref.paths import ala_dipeptide_gas_50K_reference_npz


@pytest.fixture(scope="module")
def gas_50k_data_root():
    try:
        return ensure_gas_50k_data()
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"gas-50K regression data unavailable: {exc}")


@pytest.fixture(scope="module")
def fresean_gas_50k_reference():
    reference_npz = ala_dipeptide_gas_50K_reference_npz()
    if not reference_npz.exists():
        pytest.skip(f"missing reference file: {reference_npz}")
    return np.load(reference_npz)


@pytest.fixture(scope="module")
def fresean_gas_50k_analysis(gas_50k_data_root):
    root = gas_50k_data_root
    topol = root / "MD-gas-50K" / "topol.tpr"
    traj = root / "MD-gas-50K" / "traj.trr"
    ref = root / "harmonic-normal-modes" / "min.xyz"

    u = mda.Universe(str(topol), str(traj))
    sel = u.select_atoms("all")
    u_ref = mda.Universe(str(ref))
    u.trajectory.add_transformations(
        Align(sel, reference_positions=u_ref.atoms.positions, place_com_in_box=False),
    )

    analysis = FRESEAN(
        u,
        select="all",
        n_constraints=6,
        n_corr=500,
        dt=0.004,
        sigma=10.0,
    )
    analysis.run()
    return analysis


@pytest.mark.slow
@pytest.mark.pytutorial_ref
class TestFRESEANAlaDipeptideGas50KRegression:
    """Check FRESEAN against stored gas-50K tutorial reference data."""

    def test_run_metadata(self, fresean_gas_50k_analysis, fresean_gas_50k_reference):
        ref = fresean_gas_50k_reference
        analysis = fresean_gas_50k_analysis
        assert analysis.n_frames == int(ref["n_frames"])
        assert analysis.n_corr == int(ref["n_corr"])
        assert analysis.results.n_dof == float(ref["n_dof"])

    def test_inferred_temperature(self, fresean_gas_50k_analysis, fresean_gas_50k_reference):
        ref = fresean_gas_50k_reference
        np.testing.assert_allclose(
            fresean_gas_50k_analysis.results.avg_temperature,
            float(ref["avg_temperature"]),
            rtol=1e-5,
        )

    def test_low_frequency_peaks(self, fresean_gas_50k_analysis, fresean_gas_50k_reference):
        ref = fresean_gas_50k_reference
        freqs = fresean_gas_50k_analysis.results.freqs
        vdos = fresean_gas_50k_analysis.results.vdos_total
        peaks = low_frequency_peaks(freqs, vdos, max_freq=200.0)
        np.testing.assert_allclose(
            freqs[peaks],
            ref["low_peak_freqs"],
            rtol=1e-5,
        )

    def test_total_vdos_curve(self, fresean_gas_50k_analysis, fresean_gas_50k_reference):
        ref = fresean_gas_50k_reference
        vdos = fresean_gas_50k_analysis.results.vdos_total
        np.testing.assert_allclose(
            vdos,
            ref["vdos_total"],
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            vdos.sum(),
            float(ref["vdos_sum"]),
            rtol=1e-6,
        )

    def test_frequency_grid(self, fresean_gas_50k_analysis, fresean_gas_50k_reference):
        np.testing.assert_allclose(
            fresean_gas_50k_analysis.results.freqs,
            fresean_gas_50k_reference["freqs"],
            rtol=0,
            atol=0,
        )

    def test_eigenvalues_at_low_frequencies(
        self, fresean_gas_50k_analysis, fresean_gas_50k_reference
    ):
        ref = fresean_gas_50k_reference
        eigenvalues = fresean_gas_50k_analysis.results.eigenvalues
        np.testing.assert_allclose(
            eigenvalues[0],
            ref["eigenvalues_f0"],
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            eigenvalues[1],
            ref["eigenvalues_f1"],
            rtol=1e-6,
            atol=1e-8,
        )

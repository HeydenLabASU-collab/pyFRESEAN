import pytest
import numpy as np

from pyfresean.analysis.fresean import FRESEAN
from pyfresean.transformations import Align, Unwrap
from pyfresean.tests.utils import make_Universe


class TestFRESEAN:

    # fixtures are helpful functions that set up a test
    # See more at https://docs.pytest.org/en/stable/how-to/fixtures.html
    @pytest.fixture
    def universe(self):
        u = make_Universe(
            extras=("masses", "names"),
            size=(12, 4, 2),
            n_frames=16,
            velocities=True,
        )
        u.dimensions = np.array([50.0, 50.0, 50.0, 90.0, 90.0, 90.0])
        u.atoms.masses = np.ones(u.atoms.n_atoms)
        rng = np.random.default_rng(0)
        for i, ts in enumerate(u.trajectory):
            ts.positions += rng.normal(scale=0.1, size=ts.positions.shape)
            ts.velocities = np.sin(
                np.arange(u.atoms.n_atoms * 3).reshape(u.atoms.n_atoms, 3) + i
            )
        return u

    @pytest.mark.parametrize(
        "select, n_atoms",  # argument names
        [  # argument values in a tuple, in order
            ("all", 12),
            ("index 0:3", 4),
        ],
    )
    def test_atom_selection(self, universe, select, n_atoms):
        # `universe` here is the fixture defined above
        analysis = FRESEAN(universe, select=select, n_corr=4)
        assert analysis.atomgroup.n_atoms == n_atoms

    def test_run_produces_results(self, universe):
        analysis = FRESEAN(
            universe,
            n_corr=4,
            dt=0.004,
            sigma=10.0,
        )
        analysis.run()

        n_elements = universe.atoms.n_atoms * 3
        assert analysis.results.eigenvalues.shape == (4, n_elements)
        assert analysis.results.eigenvectors.shape == (4, n_elements, n_elements)
        assert analysis.results.corr_matrix.shape == (4, n_elements, n_elements)
        assert analysis.results.freqs.shape == (4,)
        assert analysis.results.vdos_total.shape == (4,)
        assert analysis.n_frames == 16

    def test_vdos_is_finite(self, universe):
        analysis = FRESEAN(universe, n_corr=4)
        analysis.run()
        assert np.all(np.isfinite(analysis.results.vdos_total))

    def test_with_trajectory_transformations(self, universe):
        universe.trajectory.add_transformations(
            Align(universe.atoms, place_com_in_box=False),
        )
        analysis = FRESEAN(universe, n_corr=4)
        analysis.run()
        assert np.all(np.isfinite(analysis.results.vdos_total))

    def test_invalid_lag_symmetrization(self, universe):
        with pytest.raises(ValueError, match="lag_symmetrization"):
            FRESEAN(universe, n_corr=4, lag_symmetrization="invalid")

    def test_build_windowed_lags_average(self, universe):
        analysis = FRESEAN(universe, n_corr=4, lag_symmetrization="average")
        tmp_time = np.arange(16, dtype=np.float64)
        windowed = analysis._build_windowed_lags(tmp_time, n_corr=4, n_frames=16)
        assert windowed[0] == 0.0
        assert windowed[1] == (1.0 + 15.0) / 2.0
        assert windowed[2] == (2.0 + 14.0) / 2.0
        assert windowed[3] == (3.0 + 13.0) / 2.0
        assert np.allclose(windowed[4:], windowed[3:0:-1])

    def test_build_windowed_lags_mirror(self, universe):
        analysis = FRESEAN(universe, n_corr=4, lag_symmetrization="mirror")
        tmp_time = np.arange(16, dtype=np.float64)
        windowed = analysis._build_windowed_lags(tmp_time, n_corr=4, n_frames=16)
        assert np.allclose(windowed[:4], tmp_time[:4])
        assert np.allclose(windowed[4:], tmp_time[3:0:-1])

    def test_average_symmetrization_runs(self, universe):
        analysis = FRESEAN(universe, n_corr=4, lag_symmetrization="average")
        analysis.run()
        assert np.all(np.isfinite(analysis.results.corr_matrix))
        assert np.all(np.isfinite(analysis.results.vdos_total))

    def test_invalid_n_jobs(self, universe):
        with pytest.raises(ValueError, match="n_jobs"):
            FRESEAN(universe, n_corr=4, n_jobs=0)

    def test_parallel_n_jobs_matches_serial(self, universe):
        serial = FRESEAN(universe, n_corr=4, n_jobs=1)
        parallel = FRESEAN(universe, n_corr=4, n_jobs=2)
        serial.run()
        parallel.run()
        np.testing.assert_allclose(
            serial.results.corr_matrix,
            parallel.results.corr_matrix,
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            serial.results.eigenvalues,
            parallel.results.eigenvalues,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            serial.results.eigenvectors,
            parallel.results.eigenvectors,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            serial.results.vdos_total,
            parallel.results.vdos_total,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_parallel_corr_matrix_matches_serial(self, universe):
        serial = FRESEAN(universe, n_corr=4, n_jobs=1)
        parallel = FRESEAN(universe, n_corr=4, n_jobs=1)
        serial.run()
        parallel.run()
        np.testing.assert_allclose(
            serial.results.corr_matrix,
            parallel.results.corr_matrix,
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            serial.results.eigenvalues,
            parallel.results.eigenvalues,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            serial.results.vdos_total,
            parallel.results.vdos_total,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_is_parallelizable(self, universe):
        analysis = FRESEAN(universe, n_corr=4)
        assert analysis.parallelizable

    def test_multiprocessing_run_matches_serial(self, universe):
        serial = FRESEAN(universe, n_corr=4, n_jobs=1)
        parallel = FRESEAN(universe, n_corr=4, n_jobs=1)
        serial.run()
        parallel.run(n_workers=2, backend="multiprocessing", verbose=False)
        np.testing.assert_allclose(
            serial.results.corr_matrix,
            parallel.results.corr_matrix,
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            serial.results.eigenvalues,
            parallel.results.eigenvalues,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            serial.results.vdos_total,
            parallel.results.vdos_total,
            rtol=1e-12,
            atol=1e-12,
        )

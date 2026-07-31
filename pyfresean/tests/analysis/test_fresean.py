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

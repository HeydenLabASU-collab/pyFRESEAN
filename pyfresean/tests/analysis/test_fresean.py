import pytest
import numpy as np

from pyfresean.analysis.fresean import FRESEAN
from pyfresean.benchmark_keys import (
    BENCH_T_CORR_MATRIX,
    BENCH_T_VDOS,
    BENCH_T_EIGEN,
    BENCH_T_FRESEAN_TOTAL,
    BENCH_T_READ_TRAJ,
    BENCH_T_VELOCITY_SPECTRA,
    finalize_fresean_benchmark,
    new_fresean_benchmark_timings,
)
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
        assert analysis.results.eigenvectors.shape == (
            4,
            n_elements,
            n_elements,
        )
        assert analysis.results.corr_matrix.shape == (4, n_elements, n_elements)
        assert analysis.results.freqs.shape == (4,)
        assert analysis.results.corr_freqs.shape == (4,)
        assert analysis.results.vdos_total.shape == (4,)
        assert analysis.n_frames == 16

    def test_vdos_only_skips_modes(self, universe):
        full = FRESEAN(universe, n_corr=4)
        vdos_only = FRESEAN(universe, n_corr=4)
        full.run()
        vdos_only.run(compute_modes=False)
        assert vdos_only.results.eigenvalues is None
        assert vdos_only.results.eigenvectors is None
        assert vdos_only.results.mode_freqs is None
        np.testing.assert_allclose(
            vdos_only.results.vdos_total,
            full.results.vdos_total,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_compute_flags_cannot_both_be_false(self, universe):
        analysis = FRESEAN(universe, n_corr=4)
        with pytest.raises(ValueError, match="compute_modes"):
            analysis.run(compute_modes=False, compute_vdos=False)

    def test_modes_require_full_corr_matrix(self, universe):
        analysis = FRESEAN(universe, n_corr=4)
        with pytest.raises(ValueError, match="compute_corr_matrix"):
            analysis.run(compute_modes=True, compute_corr_matrix=False)

    def test_vdos_preserved_when_corr_only_second_run(self, universe):
        analysis = FRESEAN(universe, n_corr=4)
        analysis.run(
            compute_modes=False,
            compute_corr_matrix=False,
        )
        vdos_ref = analysis.results.vdos_total.copy()
        analysis.run(
            compute_modes=True,
            compute_vdos=False,
            compute_corr_matrix=True,
            read_traj=False,
        )
        np.testing.assert_allclose(
            analysis.results.vdos_total, vdos_ref, rtol=0, atol=0
        )
        assert analysis.results.eigenvalues is not None

    def test_modes_without_vdos_still_times_norm_in_vdos_phase(self, universe):
        analysis = FRESEAN(universe, n_corr=4)
        timings = analysis.run(
            compute_modes=True,
            compute_vdos=False,
            benchmark=True,
        )
        assert timings[BENCH_T_CORR_MATRIX] > 0.0
        assert timings[BENCH_T_VDOS] > 0.0

    def test_benchmark_keeps_skipped_phases_on_read_traj_false(self, universe):
        analysis = FRESEAN(universe, n_corr=4)
        first = analysis.run(
            compute_modes=False,
            compute_corr_matrix=False,
            benchmark=True,
        )
        assert first[BENCH_T_VELOCITY_SPECTRA] > 0.0
        assert first[BENCH_T_VDOS] > 0.0
        assert first[BENCH_T_EIGEN] == 0.0

        second = analysis.run(
            compute_modes=True,
            compute_vdos=False,
            compute_corr_matrix=True,
            read_traj=False,
            benchmark=True,
        )
        assert second[BENCH_T_VELOCITY_SPECTRA] == pytest.approx(
            first[BENCH_T_VELOCITY_SPECTRA]
        )
        assert second[BENCH_T_VDOS] > 0.0
        assert second[BENCH_T_CORR_MATRIX] > 0.0
        assert second[BENCH_T_EIGEN] > 0.0
        assert second[BENCH_T_FRESEAN_TOTAL] == pytest.approx(
            second[BENCH_T_READ_TRAJ]
            + second[BENCH_T_VELOCITY_SPECTRA]
            + second[BENCH_T_CORR_MATRIX]
            + second[BENCH_T_VDOS]
            + second[BENCH_T_EIGEN]
        )

    def test_benchmark_overwrites_rerun_phase_not_skipped(self, universe):
        analysis = FRESEAN(universe, n_corr=4)
        analysis.run(compute_modes=False, compute_corr_matrix=False)
        bogus = new_fresean_benchmark_timings()
        bogus[BENCH_T_VELOCITY_SPECTRA] = 999.0
        bogus[BENCH_T_CORR_MATRIX] = 999.0
        analysis.benchmark = finalize_fresean_benchmark(bogus)

        timings = analysis.run(
            compute_modes=True,
            compute_vdos=False,
            compute_corr_matrix=True,
            read_traj=False,
            benchmark=True,
        )
        assert timings[BENCH_T_VELOCITY_SPECTRA] == 999.0
        assert 0.0 < timings[BENCH_T_CORR_MATRIX] < 999.0
        assert timings[BENCH_T_EIGEN] > 0.0

    def test_vdos_diagonal_path_matches_full_matrix(self, universe):
        full = FRESEAN(universe, n_corr=4)
        fast = FRESEAN(universe, n_corr=4)
        full.run(compute_modes=False, compute_corr_matrix=True)
        fast.run(
            compute_modes=False,
            compute_corr_matrix=False,
        )
        assert fast.results.corr_matrix is None
        assert fast.results.corr_freqs is None
        np.testing.assert_allclose(
            fast.results.vdos_total,
            full.results.vdos_total,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            fast.results.avg_temperature,
            full.results.avg_temperature,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_mode_freq_subset(self, universe):
        analysis = FRESEAN(universe, n_corr=4, dt=0.004)
        target = float(analysis._frequency_grid(4, 0.004)[2])
        subset = FRESEAN(universe, n_corr=4, dt=0.004)
        full = FRESEAN(universe, n_corr=4, dt=0.004)
        full.run()
        subset.run(compute_modes=[target], compute_vdos=[target])
        n_elements = universe.atoms.n_atoms * 3
        assert subset.results.corr_matrix.shape == (4, n_elements, n_elements)
        assert subset.results.eigenvalues.shape == (1, n_elements)
        assert subset.results.vdos_total.shape == (1,)
        np.testing.assert_allclose(
            subset.results.corr_matrix[2],
            full.results.corr_matrix[2],
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            subset.results.eigenvalues[0],
            full.results.eigenvalues[2],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_empty_frequency_list_raises(self, universe):
        analysis = FRESEAN(universe, n_corr=4)
        with pytest.raises(ValueError, match="empty"):
            analysis.run(compute_vdos=[])

    def test_reuse_corr_matrix_for_modes_after_vdos(self, universe):
        analysis = FRESEAN(universe, n_corr=4)
        analysis.run(compute_modes=False)
        assert analysis.results.eigenvalues is None
        analysis.run(compute_modes=True, read_traj=False)
        n_elements = universe.atoms.n_atoms * 3
        assert analysis.results.eigenvalues.shape == (4, n_elements)

    def test_modes_from_velocity_cache(self, universe):
        analysis = FRESEAN(universe, n_corr=4)
        analysis.run(compute_modes=False)
        analysis.results.corr_matrix = None
        analysis.run(compute_modes=True, compute_vdos=False, read_traj=False)
        n_elements = universe.atoms.n_atoms * 3
        assert analysis.results.eigenvalues.shape == (4, n_elements)

    def test_vdos_freqs_nearest_grid_bin(self, universe):
        grid = FRESEAN._frequency_grid(4, 0.004)
        between = float((grid[1] + grid[2]) / 2.0)
        analysis = FRESEAN(
            universe,
            n_corr=4,
            dt=0.004,
        )
        analysis.run(compute_modes=False, compute_vdos=[between])
        assert analysis.results.vdos_freqs.shape == (1,)
        assert analysis.results.vdos_freqs[0] in grid

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
        windowed = analysis._build_windowed_lags(
            tmp_time, n_corr=4, n_frames=16
        )
        assert windowed[0] == 0.0
        assert windowed[1] == (1.0 + 15.0) / 2.0
        assert windowed[2] == (2.0 + 14.0) / 2.0
        assert windowed[3] == (3.0 + 13.0) / 2.0
        assert np.allclose(windowed[4:], windowed[3:0:-1])

    def test_build_windowed_lags_mirror(self, universe):
        analysis = FRESEAN(universe, n_corr=4, lag_symmetrization="mirror")
        tmp_time = np.arange(16, dtype=np.float64)
        windowed = analysis._build_windowed_lags(
            tmp_time, n_corr=4, n_frames=16
        )
        assert np.allclose(windowed[:4], tmp_time[:4])
        assert np.allclose(windowed[4:], tmp_time[3:0:-1])

    def test_average_symmetrization_runs(self, universe):
        analysis = FRESEAN(universe, n_corr=4, lag_symmetrization="average")
        analysis.run()
        assert np.all(np.isfinite(analysis.results.corr_matrix))
        assert np.all(np.isfinite(analysis.results.vdos_total))

    def test_invalid_parallel_n_jobs(self, universe):
        with pytest.raises(ValueError, match="n_jobs"):
            FRESEAN(
                universe,
                n_corr=4,
                parallel={"corr_matrix": {"n_jobs": 0}},
            )

    def test_invalid_parallel_phase(self, universe):
        with pytest.raises(ValueError, match="unknown parallel phase"):
            FRESEAN(
                universe,
                n_corr=4,
                parallel={"not_a_phase": {"n_jobs": 1}},
            )

    def test_parallel_per_phase_settings(self, universe):
        analysis = FRESEAN(
            universe,
            n_corr=4,
            parallel={
                "corr_matrix": {"n_jobs": 4, "omp_threads": 1},
                "eigen": {"n_jobs": 1, "omp_threads": 1},
            },
        )
        assert analysis._parallel["corr_matrix"].n_jobs == 4
        assert analysis._parallel["eigen"].n_jobs == 1
        assert analysis._parallel["velocity_fft"].n_jobs == 1

    def test_corr_matrix_tiles_cover_upper_triangle(self):
        budget_pairs = 6
        tiles = FRESEAN._corr_matrix_tiles(5, budget_pairs)
        covered = set()
        for i0, i1, j0, j1 in tiles:
            for i in range(i0, i1):
                for j in range(j0, j1):
                    if j >= i:
                        covered.add((i, j))
        expected = {
            (i, j) for i in range(5) for j in range(i, 5)
        }
        assert covered == expected
        assert all(j0 >= i0 for i0, _, j0, _ in tiles)

    def test_parallel_n_jobs_matches_serial(self, universe):
        serial = FRESEAN(universe, n_corr=4)
        parallel = FRESEAN(
            universe,
            n_corr=4,
            parallel={"corr_matrix": {"n_jobs": 2, "omp_threads": 1}},
        )
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
        serial = FRESEAN(universe, n_corr=4)
        parallel = FRESEAN(universe, n_corr=4)
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

    def test_run_benchmark_returns_phase_timings(self, universe):
        analysis = FRESEAN(universe, n_corr=4)
        timings = analysis.run(benchmark=True)

        assert set(timings) == {
            BENCH_T_READ_TRAJ,
            BENCH_T_VELOCITY_SPECTRA,
            BENCH_T_CORR_MATRIX,
            BENCH_T_VDOS,
            BENCH_T_EIGEN,
            BENCH_T_FRESEAN_TOTAL,
        }
        assert analysis.benchmark == timings
        for key in (
            BENCH_T_READ_TRAJ,
            BENCH_T_VELOCITY_SPECTRA,
            BENCH_T_CORR_MATRIX,
            BENCH_T_VDOS,
            BENCH_T_EIGEN,
            BENCH_T_FRESEAN_TOTAL,
        ):
            assert timings[key] >= 0.0
        assert timings[BENCH_T_VDOS] > 0.0
        assert timings[BENCH_T_READ_TRAJ] > 0.0
        assert timings[BENCH_T_FRESEAN_TOTAL] == pytest.approx(
            timings[BENCH_T_READ_TRAJ]
            + timings[BENCH_T_VELOCITY_SPECTRA]
            + timings[BENCH_T_CORR_MATRIX]
            + timings[BENCH_T_VDOS]
            + timings[BENCH_T_EIGEN]
        )
        assert np.all(np.isfinite(analysis.results.vdos_total))

    def test_multiprocessing_run_matches_serial(self, universe):
        serial = FRESEAN(universe, n_corr=4)
        parallel = FRESEAN(universe, n_corr=4)
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

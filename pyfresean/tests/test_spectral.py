import numpy as np
import pytest

from pyfresean.postprocess.spectral import (
    cartesian_displacements,
    cluster_mode_indices,
    low_frequency_peaks,
    mode_overlap_matrix,
    mode_spectrum,
    mode_spectra,
    peak_indices,
    select_modes_in_frequency_range,
    sort_clusters_by_peak_frequency,
    total_vdos,
    weighted_similarity_matrix,
)


def test_total_vdos_and_peaks():
    eigenvalues = np.array([[1.0, 0.5, 0.0], [0.2, 0.1, 0.0], [2.0, 0.0, 0.0]])
    vdos = total_vdos(eigenvalues)
    np.testing.assert_allclose(vdos, [1.5, 0.3, 2.0])

    vdos_with_peak = np.array([0.1, 2.0, 0.2])
    peaks = peak_indices(vdos_with_peak)
    assert list(peaks) == [1]


def test_mode_spectrum():
    corr = np.array([np.eye(3), 2 * np.eye(3)])
    mode = np.array([1.0, 0.0, 0.0])
    spectrum = mode_spectrum(corr, mode)
    np.testing.assert_allclose(spectrum, [1.0, 2.0])


def test_mode_spectra_and_low_frequency_peaks():
    corr = np.array([np.eye(2), 2 * np.eye(2)])
    modes = np.array([[1.0, 0.0], [0.0, 1.0]])
    spectra = mode_spectra(corr, modes)
    assert spectra.shape == (2, 2)

    freqs = np.array([0.0, 10.0, 20.0, 30.0])
    vdos = np.array([0.1, 2.0, 1.0, 0.2])
    peaks = low_frequency_peaks(freqs, vdos, max_freq=25.0)
    assert list(peaks) == [1]


def test_clustering_helpers():
    freqs = np.linspace(0, 100, 5)
    eigenvalues = np.array([[1.0, 0.5, 0.1, 0.0, 0.0]] * 5)
    eigenvectors = np.tile(np.eye(5), (5, 1, 1))

    vec_sel, val_sel = select_modes_in_frequency_range(
        freqs, eigenvalues, eigenvectors, max_freq=100.0, n_modes_per_freq=2
    )
    assert vec_sel.shape == (8, 5)
    assert val_sel.shape == (8,)

    cluster_matrix = weighted_similarity_matrix(val_sel, vec_sel)
    assert cluster_matrix.shape == (8, 8)
    np.testing.assert_allclose(np.diag(cluster_matrix), np.diag(cluster_matrix))

    cluster_idx = cluster_mode_indices(cluster_matrix, cutoff=0.3)
    assert len(cluster_idx) >= 1

    cluster_vdos = np.array([[0.1, 0.2, 0.9, 0.1], [0.1, 0.8, 0.2, 0.1]])
    sort_idx, sorted_peaks = sort_clusters_by_peak_frequency(cluster_vdos)
    assert list(sort_idx) == [1, 0]
    assert list(sorted_peaks) == [1, 2]

    overlap = mode_overlap_matrix(np.eye(3))
    np.testing.assert_allclose(overlap, np.eye(3))


def test_cartesian_displacements_divides_by_sqrt_mass():
    masses = np.array([1.0, 4.0, 16.0])
    mode = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    got = cartesian_displacements(mode, masses)
    expected = mode / np.sqrt(masses)[:, np.newaxis]
    np.testing.assert_allclose(got, expected)


def test_cartesian_displacements_accepts_bead_masses_and_flat_vector():
    masses = np.array([14.0, 36.0])
    flat = np.arange(6, dtype=np.float64)
    got = cartesian_displacements(flat, masses, scale=10.0)
    expected = 10.0 * flat.reshape(2, 3) / np.sqrt(masses)[:, np.newaxis]
    np.testing.assert_allclose(got, expected)


def test_cartesian_displacements_rejects_bad_input():
    with pytest.raises(ValueError, match="positive"):
        cartesian_displacements(np.ones((2, 3)), np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="shape"):
        cartesian_displacements(np.ones((3, 3)), np.array([1.0, 4.0]))

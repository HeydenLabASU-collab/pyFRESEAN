from __future__ import annotations

import numpy as np
from scipy.fft import ifft


def total_vdos(eigenvalues: np.ndarray) -> np.ndarray:
    return np.sum(eigenvalues, axis=1)


def peak_indices(vdos: np.ndarray) -> np.ndarray:
    peaks = np.where((vdos[1:-1] > vdos[:-2]) & (vdos[1:-1] > vdos[2:]))[0] + 1
    return peaks


def vdos_peaks(vdos: np.ndarray, extra: tuple[int, ...] = ()) -> np.ndarray:
    peaks = peak_indices(vdos)
    if extra:
        peaks = np.sort(
            np.unique(np.concatenate([np.asarray(extra, dtype=int), peaks]))
        )
    return peaks


def low_frequency_peaks(
    freqs: np.ndarray,
    vdos: np.ndarray,
    max_freq: float = 200.0,
    extra: tuple[int, ...] = (),
) -> np.ndarray:
    peaks = vdos_peaks(vdos, extra=extra)
    return peaks[freqs[peaks] < max_freq]


def mode_spectrum(corr_matrix: np.ndarray, mode: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.dot(mode, np.dot(corr_matrix[i], mode))
            for i in range(corr_matrix.shape[0])
        ]
    )


def mode_spectra(
    corr_matrix: np.ndarray,
    modes: np.ndarray,
) -> np.ndarray:
    return np.array([mode_spectrum(corr_matrix, mode) for mode in modes])


def mode_time_correlation(
    corr_matrix: np.ndarray,
    mode_i: np.ndarray,
    mode_j: np.ndarray,
    n_corr: int,
    win_time: np.ndarray,
    avg_temp: float,
) -> tuple[np.ndarray, np.ndarray]:
    tmp1 = np.zeros(2 * n_corr - 1, dtype=np.float64)
    for k in range(n_corr):
        tmp1[k] = np.dot(mode_i, np.dot(corr_matrix[k], mode_j))
    tmp1[n_corr:] = tmp1[n_corr - 1 : 0 : -1]
    vcf = np.real(ifft(tmp1))[:n_corr] * avg_temp * n_corr
    vcf_actual = vcf / win_time[:n_corr] * win_time[0]
    return vcf, vcf_actual


def select_modes_in_frequency_range(
    freqs: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    max_freq: float = 200.0,
    min_freq: float = 0.0,
    n_modes_per_freq: int = 5,
    exclude_zero_freq: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if exclude_zero_freq:
        mask = (freqs > 0) & (freqs < max_freq + 1)
    else:
        mask = (freqs >= min_freq) & (freqs < max_freq + 1)
    freq_sel = np.where(mask)[0]
    eigenvectors_sel = []
    eigenvalues_sel = []
    for freq_idx in freq_sel:
        for mode_idx in range(n_modes_per_freq):
            eigenvalues_sel.append(eigenvalues[freq_idx, mode_idx])
            eigenvectors_sel.append(eigenvectors[freq_idx, mode_idx])
    return np.asarray(eigenvectors_sel), np.asarray(eigenvalues_sel)


def _subtract_rigid_overlap(
    mode: np.ndarray,
    rigid_modes: np.ndarray,
) -> np.ndarray:
    vec = mode.copy()
    for rigid_mode in rigid_modes:
        vec -= np.dot(vec, rigid_mode) * rigid_mode
    return vec


def weighted_similarity_matrix(
    eigenvalues_sel: np.ndarray,
    eigenvectors_sel: np.ndarray,
    rigid_modes: np.ndarray | None = None,
) -> np.ndarray:
    n_sel = len(eigenvalues_sel)
    max_eigenvalue = np.max(eigenvalues_sel)
    cluster_matrix = np.zeros((n_sel, n_sel), dtype=np.float64)
    if max_eigenvalue <= 0:
        return cluster_matrix

    for i in range(n_sel):
        a = np.sqrt(eigenvalues_sel[i]) if eigenvalues_sel[i] > 0 else 0.0
        vec_i = eigenvectors_sel[i]
        if rigid_modes is not None:
            vec_i = _subtract_rigid_overlap(vec_i, rigid_modes)
        for j in range(n_sel):
            b = np.sqrt(eigenvalues_sel[j]) if eigenvalues_sel[j] > 0 else 0.0
            vec_j = eigenvectors_sel[j]
            if rigid_modes is not None:
                vec_j = _subtract_rigid_overlap(vec_j, rigid_modes)
            cluster_matrix[i, j] = (
                a * b / max_eigenvalue * np.abs(np.dot(vec_i, vec_j))
            )
    return cluster_matrix


def cluster_mode_indices(
    cluster_matrix: np.ndarray,
    cutoff: float = 0.3,
) -> np.ndarray:
    cluster_indices: list[int] = []
    matrix = cluster_matrix.copy()
    if np.max(matrix) <= cutoff:
        return np.array(cluster_indices, dtype=int)

    while True:
        row_counts = np.sum(matrix > cutoff, axis=1)
        max_count = np.max(row_counts)
        if max_count == 0:
            break
        max_row_index = int(np.where(row_counts == max_count)[0][-1])
        cluster_indices.append(max_row_index)
        indices_to_remove = np.where(matrix[max_row_index] > cutoff)[0]
        matrix[:, indices_to_remove] = 0
        matrix[indices_to_remove, :] = 0
    return np.asarray(cluster_indices, dtype=int)


def sort_clusters_by_peak_frequency(
    cluster_vdos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    peak_indices = np.argmax(cluster_vdos, axis=1)
    sort_indices = np.argsort(peak_indices)
    return sort_indices, peak_indices[sort_indices]


def mode_overlap_matrix(modes: np.ndarray) -> np.ndarray:
    return np.abs(modes @ modes.T)

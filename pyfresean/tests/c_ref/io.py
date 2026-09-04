"""Read FRESEAN COARSE (C) binary and text output files."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def load_c_eigenvalues_dat(path: Path | str) -> np.ndarray:
    """
    Load ``eval_covar_<stem>.mmat.dat`` from ``fresean eigen``.

    Returns an array of shape ``(n_freq, n_dof)`` with eigenvalues sorted
    large-to-small per frequency row (same order as C writes after ``eigen``).
    """
    path = Path(path)
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
    if not rows:
        raise ValueError(f"no eigenvalue rows in {path}")
    return np.asarray(rows, dtype=np.float64)


def load_c_evec_mmat(path: Path | str, n_corr: int | None = None) -> np.ndarray:
    """
    Load ``evec_covar_<stem>.mmat`` (binary eigenvectors from ``fresean eigen``).

    Returns ``(n_freq, n_dof, n_dof)`` with eigenvectors as rows (C stores
    transposed matrices: each row is one eigenvector).
    """
    path = Path(path)
    with path.open("rb") as handle:
        magic1 = struct.unpack("i", handle.read(4))[0]
        file_n_corr = struct.unpack("i", handle.read(4))[0]
        n1 = struct.unpack("i", handle.read(4))[0]
        n2 = struct.unpack("i", handle.read(4))[0]
        magic1_end = struct.unpack("i", handle.read(4))[0]
        if magic1 != magic1_end:
            raise ValueError(f"header magic mismatch in {path}")
        if n1 != n2:
            raise ValueError(f"non-square eigenvector matrices in {path}")

        n_freq = n_corr if n_corr is not None else file_n_corr
        if n_corr is not None and file_n_corr != n_corr:
            raise ValueError(
                f"expected n_corr={n_corr}, file header has {file_n_corr}"
            )

        block_bytes = n1 * n2 * 8
        vectors = np.empty((n_freq, n1, n1), dtype=np.float64)
        for freq_index in range(n_freq):
            magic_block = struct.unpack("i", handle.read(4))[0]
            if magic_block != block_bytes:
                raise ValueError(f"block size mismatch at frequency {freq_index}")
            raw = struct.unpack(f"{n1 * n1}d", handle.read(block_bytes))
            vectors[freq_index] = np.asarray(raw, dtype=np.float64).reshape(n1, n1)
            magic_block_end = struct.unpack("i", handle.read(4))[0]
            if magic_block_end != block_bytes:
                raise ValueError(f"block footer mismatch at frequency {freq_index}")
    return vectors


def max_abs_eigenvector_correlation(
    py_vectors: np.ndarray,
    c_vectors: np.ndarray,
    freq_index: int,
    n_modes: int,
) -> float:
    """
    Largest absolute correlation between pyfresean and C eigenvector rows
    at one frequency (accounts for sign flips).
    """
    py = py_vectors[freq_index, :n_modes]
    c = c_vectors[freq_index, :n_modes]
    corr = np.abs(py @ c.T)
    return float(np.max(corr))

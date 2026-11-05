"""Symmetry-adapted basis S and block-wise diagonalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pyfresean.analysis.symmetry.groups import (
    resolve_partner_map,
    symmetry_order_from_label,
)


@dataclass(frozen=True)
class SymmetryBasis:
    """
    Orthogonal change of basis for cyclic symmetry.

    DOFs are permuted to slot-contiguous layout (``dof_ordering``) before
    applying block-diagonal ``S``.
    """

    label: str
    order: int
    partner_map: np.ndarray
    dof_ordering: np.ndarray
    S: np.ndarray
    block_slices: tuple[tuple[int, int], ...]


def _build_slot_generator(order: int) -> np.ndarray:
    """
    One C_n copy step on a motif slot (``3 * order`` DOFs).

    Copies are permuted in fixed lab (x, y, z); no 3D rotation matrix.
    """
    d = 3 * order
    mat = np.zeros((d, d), dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for a in range(order):
        ib = (a + 1) % order
        mat[3 * ib : 3 * ib + 3, 3 * a : 3 * a + 3] = eye
    return mat


def _orthonormal_columns(matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if matrix.size == 0:
        return matrix.reshape(matrix.shape[0], 0)
    q, _ = np.linalg.qr(matrix)
    keep = [i for i in range(q.shape[1]) if np.linalg.norm(q[:, i]) > tol]
    if not keep:
        return q[:, :0]
    return q[:, keep]


def _real_1d_basis_from_projector(projector: np.ndarray) -> np.ndarray:
    sym = (projector + projector.T) / 2.0
    evals, evecs = np.linalg.eigh(sym)
    cols = [vec for val, vec in zip(evals, evecs.T) if val > 0.5]
    if not cols:
        return np.zeros((projector.shape[0], 0))
    return _orthonormal_columns(np.column_stack(cols))


def _real_basis_from_complex_projector(projector: np.ndarray) -> np.ndarray:
    """All orthonormal real modes in the image of a complex irrep projector."""
    evals, evecs = np.linalg.eigh(projector)
    parts: list[np.ndarray] = []
    for val, vec in zip(evals, evecs.T):
        if val < 0.5:
            continue
        w = vec / (np.linalg.norm(vec) + 1e-15)
        parts.append(np.real(w))
        parts.append(np.imag(w))
    if not parts:
        return np.zeros((projector.shape[0], 0))
    return _orthonormal_columns(np.column_stack(parts))


def _symmetry_basis_for_slot(
    m_slot: np.ndarray, order: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    powers = [np.linalg.matrix_power(m_slot, k) for k in range(order)]
    omega = np.exp(2j * np.pi / order)

    columns: list[np.ndarray] = []
    block_slices: list[tuple[int, int]] = []
    cursor = 0

    p0 = np.real(sum(powers) / order)
    b0 = _real_1d_basis_from_projector(p0)
    if b0.shape[1] > 0:
        columns.append(b0)
        block_slices.append((cursor, cursor + b0.shape[1]))
        cursor += b0.shape[1]

    if order % 2 == 0:
        alt = np.real(
            sum(((-1) ** k) * powers[k] for k in range(order)) / order
        )
        bh = _real_1d_basis_from_projector(alt)
        if bh.shape[1] > 0:
            columns.append(bh)
            block_slices.append((cursor, cursor + bh.shape[1]))
            cursor += bh.shape[1]

    for j in range(1, order):
        if order % 2 == 0 and j == order // 2:
            continue
        if j > order - j:
            continue
        pj = sum(omega ** (-j * k) * powers[k] for k in range(order)) / order
        bj = _real_basis_from_complex_projector(pj)
        if bj.shape[1] > 0:
            columns.append(bj)
            block_slices.append((cursor, cursor + bj.shape[1]))
            cursor += bj.shape[1]

    if not columns:
        raise RuntimeError("failed to build symmetry-adapted basis for slot")

    d = m_slot.shape[0]
    s_part = _orthonormal_columns(np.hstack(columns))
    k = s_part.shape[1]
    if k < d:
        aug = np.hstack([s_part, np.eye(d, dtype=np.float64)])
        s_slot, _ = np.linalg.qr(aug)
        s_slot = s_slot[:, :d]
        block_slices.append((k, d))
    elif k == d:
        s_slot = s_part
    else:
        raise RuntimeError(
            f"symmetry basis has too many columns ({k}) for dimension {d}"
        )
    return s_slot, block_slices


def _dof_ordering_for_slots(partner_map: np.ndarray, order: int) -> np.ndarray:
    parts: list[int] = []
    m_slots = partner_map.shape[0]
    for k in range(m_slots):
        for copy_j in range(order):
            atom_i = int(partner_map[k, copy_j])
            parts.extend(range(3 * atom_i, 3 * atom_i + 3))
    ordering = np.array(parts, dtype=np.intp)
    if len(np.unique(ordering)) != ordering.size:
        raise RuntimeError("invalid DOF ordering for symmetry slots")
    return ordering


def build_symmetry_basis(
    atomgroup,
    symmetry_label: str,
    symmetry_groups: Sequence,
) -> SymmetryBasis:
    """Build ``S`` and block slices for ``C3`` / ``C6`` from copy groups only."""
    order = symmetry_order_from_label(symmetry_label)
    partner_map = resolve_partner_map(atomgroup, symmetry_groups, order)
    dof_ordering = _dof_ordering_for_slots(partner_map, order)
    n_dof = dof_ordering.size
    slot_dim = 3 * order
    m_slots = partner_map.shape[0]
    if n_dof != m_slots * slot_dim:
        raise RuntimeError("symmetry DOF count mismatch")

    s_full = np.zeros((n_dof, n_dof), dtype=np.float64)
    block_slices: list[tuple[int, int]] = []

    for k in range(m_slots):
        m_slot = _build_slot_generator(order)
        s_slot, slot_blocks = _symmetry_basis_for_slot(m_slot, order)
        base = k * slot_dim
        s_full[base : base + slot_dim, base : base + slot_dim] = s_slot
        for start, end in slot_blocks:
            block_slices.append((base + start, base + end))

    if not np.allclose(s_full.T @ s_full, np.eye(n_dof), atol=1e-8):
        raise RuntimeError("symmetry basis S is not orthonormal")

    return SymmetryBasis(
        label=symmetry_label.strip().upper(),
        order=order,
        partner_map=partner_map,
        dof_ordering=dof_ordering,
        S=s_full,
        block_slices=tuple(block_slices),
    )


def _fill_eigen_one_bin(
    corr: np.ndarray,
    basis: SymmetryBasis,
    n_elements: int,
) -> tuple[np.ndarray, np.ndarray]:
    ordering = basis.dof_ordering
    corr_perm = corr[np.ix_(ordering, ordering)]
    corr_prime = basis.S.T @ corr_perm @ basis.S
    n_perm = ordering.size
    eigenvalues = np.empty(n_elements, dtype=np.float64)
    eigenvectors = np.empty((n_elements, n_elements), dtype=np.float64)

    off_diag_max = 0.0
    slices = list(basis.block_slices)
    for i, (a0, a1) in enumerate(slices):
        for j, (b0, b1) in enumerate(slices):
            if i == j:
                continue
            off_diag_max = max(
                off_diag_max,
                float(np.max(np.abs(corr_prime[a0:a1, b0:b1]))),
            )

    if off_diag_max < 1e-8:
        modes_perm: list[np.ndarray] = []
        mode_vals: list[float] = []
        for start, end in slices:
            block = corr_prime[start:end, start:end]
            vals, vecs = np.linalg.eigh(block)
            for val, vec in zip(vals, vecs.T):
                full_sym = np.zeros(n_perm, dtype=np.float64)
                full_sym[start:end] = vec
                modes_perm.append(full_sym)
                mode_vals.append(float(val))
        v_sym = np.column_stack(modes_perm)
        sort_idx = np.argsort(mode_vals)[::-1]
        eigenvalues[:n_perm] = np.asarray(mode_vals, dtype=np.float64)[sort_idx]
        v_sym = v_sym[:, sort_idx]
    else:
        vals, vecs = np.linalg.eigh(corr_prime)
        order = np.argsort(vals)[::-1]
        eigenvalues[:n_perm] = vals[order]
        v_sym = vecs[:, order]

    v_perm = basis.S @ v_sym
    inv_order = np.empty_like(ordering)
    inv_order[ordering] = np.arange(n_perm, dtype=np.intp)
    v_orig = v_perm[inv_order, :]
    eigenvectors[:, :n_perm] = v_orig.T
    if n_perm < n_elements:
        eigenvalues[n_perm:] = 0.0
    return eigenvalues, eigenvectors


def diagonalize_corr_with_symmetry(
    corr_matrix: np.ndarray,
    freq_indices: np.ndarray,
    basis: SymmetryBasis,
    n_elements: int,
    *,
    n_jobs: int = 1,
    omp_threads: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Block-wise diagonalization using precomputed symmetry basis."""
    from concurrent.futures import ThreadPoolExecutor

    from pyfresean.parallel import blas_thread_context, resolve_n_jobs

    n_bins = int(freq_indices.size)
    eigenvalues = np.empty((n_bins, n_elements), dtype=np.float64)
    eigenvectors = np.empty((n_bins, n_elements, n_elements), dtype=np.float64)

    def work(out_index: int, freq_index: int) -> None:
        ev, evec = _fill_eigen_one_bin(
            corr_matrix[freq_index], basis, n_elements
        )
        eigenvalues[out_index] = ev
        eigenvectors[out_index] = evec

    max_workers = resolve_n_jobs(n_jobs)
    with blas_thread_context(omp_threads):
        if max_workers == 1:
            for out_index, freq_index in enumerate(freq_indices):
                work(out_index, int(freq_index))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(
                    executor.map(
                        lambda pair: work(pair[0], int(pair[1])),
                        enumerate(freq_indices),
                    )
                )
    return eigenvalues, eigenvectors

"""Tests for symmetry-adapted mode diagonalization."""

import numpy as np
import pytest

from pyfresean.analysis.symmetry.basis import (
    _build_slot_generator,
    _symmetry_basis_for_slot,
)
from pyfresean.analysis.fresean import FRESEAN
from pyfresean.tests.utils import make_Universe


def _twirled_symmetric(m_slot: np.ndarray, order: int, rng: np.random.Generator) -> np.ndarray:
    powers = [np.linalg.matrix_power(m_slot, k) for k in range(order)]
    a = rng.standard_normal((m_slot.shape[0], m_slot.shape[0]))
    a = (a + a.T) / 2.0
    c = sum(p.T @ a @ p for p in powers) / order
    return (c + c.T) / 2.0


class TestSymmetryBasis:
    @pytest.mark.parametrize("order", [3, 6])
    def test_slot_basis_orthonormal_and_block_diag(self, order):
        m_slot = _build_slot_generator(order)
        s_slot, block_slices = _symmetry_basis_for_slot(m_slot, order)
        d = 3 * order
        assert s_slot.shape == (d, d)
        assert np.allclose(s_slot.T @ s_slot, np.eye(d), atol=1e-9)
        rng = np.random.default_rng(0)
        c = _twirled_symmetric(m_slot, order, rng)
        c_prime = s_slot.T @ c @ s_slot
        off = 0.0
        for i, (a0, a1) in enumerate(block_slices):
            for j, (b0, b1) in enumerate(block_slices):
                if i == j:
                    continue
                off = max(off, float(np.max(np.abs(c_prime[a0:a1, b0:b1]))))
        assert off < 1e-9

    @pytest.mark.parametrize("order", [3, 6])
    def test_eigenvalues_match_full_eigh(self, order):
        m_slot = _build_slot_generator(order)
        s_slot, block_slices = _symmetry_basis_for_slot(m_slot, order)
        rng = np.random.default_rng(1)
        c = _twirled_symmetric(m_slot, order, rng)
        vals_full = np.linalg.eigh(c)[0]
        c_prime = s_slot.T @ c @ s_slot
        vals_blocks = []
        for start, end in block_slices:
            vals_blocks.append(np.linalg.eigh(c_prime[start:end, start:end])[0])
        vals_sym = np.concatenate(vals_blocks)
        assert np.allclose(np.sort(vals_full), np.sort(vals_sym), atol=1e-9)


def _c3_triangle_universe():
    u = make_Universe(
        extras=("masses",),
        size=(3, 1, 1),
        n_frames=4,
        velocities=True,
    )
    r = 1.0
    angles = np.deg2rad([0.0, 120.0, 240.0])
    u.atoms.positions = np.column_stack(
        [r * np.cos(angles), r * np.sin(angles), np.zeros(3)]
    )
    u.atoms.masses = np.ones(3)
    for ts in u.trajectory:
        ts.velocities = np.random.default_rng(2).normal(size=(3, 3))
    return u


class TestFRESEANSymmetry:
    def test_c3_run_matches_full_diagonalization_eigenvalues(self):
        u = _c3_triangle_universe()
        groups = (u.atoms[0:1], u.atoms[1:2], u.atoms[2:3])
        full = FRESEAN(u, select="all", n_corr=4, n_constraints=0)
        sym = FRESEAN(
            u,
            select="all",
            n_corr=4,
            n_constraints=0,
            symmetry="C3",
            symmetry_groups=groups,
        )
        full.run()
        sym.run()
        assert np.allclose(
            np.sort(full.results.eigenvalues, axis=1),
            np.sort(sym.results.eigenvalues, axis=1),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_invalid_group_partition_raises(self):
        u = _c3_triangle_universe()
        with pytest.raises(ValueError, match="not in the FRESEAN selection"):
            FRESEAN(
                u,
                select="index 0 1",
                n_corr=4,
                symmetry="C3",
                symmetry_groups=(
                    u.select_atoms("index 0"),
                    u.select_atoms("index 1"),
                    u.atoms[2:3],
                ),
            )

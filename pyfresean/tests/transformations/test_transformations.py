import warnings

import numpy as np
import pytest

from pyfresean.exceptions import MissingBoxWarning
from pyfresean.transformations import Align, Unwrap
from pyfresean.tests.utils import make_Universe


def test_align_warns_and_uses_dummy_box_when_wrapping():
    u = make_Universe(extras=("masses",), n_frames=1)
    u.atoms.masses = np.ones(u.atoms.n_atoms)
    u.dimensions = None
    align = Align(u.atoms, place_com_in_box=True)

    with pytest.warns(MissingBoxWarning, match="dummy box"):
        align._ensure_box()

    assert u.dimensions is not None


def test_align_no_warning_without_wrapping():
    u = make_Universe(extras=("masses",), n_frames=1)
    u.atoms.masses = np.ones(u.atoms.n_atoms)
    u.dimensions = None
    align = Align(u.atoms, place_com_in_box=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        align(u.trajectory[0])

    assert not any(issubclass(w.category, MissingBoxWarning) for w in caught)
    assert u.dimensions is None


def test_unwrap_warns_and_skips_without_box():
    u = make_Universe(n_frames=1)
    u.dimensions = None
    unwrap = Unwrap(u.atoms)

    with pytest.warns(MissingBoxWarning, match="skipping unwrap"):
        ts = unwrap(u.trajectory[0])

    assert ts is u.trajectory[0]


def test_align_superimposes_translated_copy():
    u = make_Universe(extras=("masses",), n_frames=1)
    u.atoms.masses = np.ones(u.atoms.n_atoms)
    ref = u.atoms.positions.copy() + np.array([10.0, 0.0, 0.0])
    align = Align(u.atoms, reference_positions=ref, place_com_in_box=False)
    align(u.trajectory[0])
    rmsd = np.sqrt(np.mean(np.sum((u.atoms.positions - ref) ** 2, axis=1)))
    assert rmsd < 1e-6


def test_subtract_com_velocity():
    u = make_Universe(extras=("masses",), n_frames=1, velocities=True)
    u.atoms.masses = np.ones(u.atoms.n_atoms)
    u.atoms.velocities = np.random.default_rng(0).standard_normal(
        (u.atoms.n_atoms, 3)
    )
    align = Align(u.atoms, place_com_in_box=False, subtract_com_velocity=True)
    align(u.trajectory[0])
    sel = u.atoms
    com_vel = np.average(sel.velocities, axis=0, weights=sel.masses)
    np.testing.assert_allclose(com_vel, 0, atol=1e-6)


def test_add_transformations():
    u = make_Universe(extras=("masses",), n_frames=2, velocities=True)
    u.dimensions = np.array([50.0, 50.0, 50.0, 90.0, 90.0, 90.0])
    u.atoms.masses = np.ones(u.atoms.n_atoms)

    u.trajectory.add_transformations(
        Unwrap(u.atoms),
        Align(u.atoms, place_com_in_box=False),
    )
    assert len(u.trajectory.transformations) == 2

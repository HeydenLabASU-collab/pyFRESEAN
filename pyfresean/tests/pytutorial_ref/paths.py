"""Paths for pyfresean vs FRESEAN Python tutorial reference comparison tests."""

from __future__ import annotations

from pathlib import Path


def _project_root() -> Path:
    """``pyfresean/`` project directory (contains ``examples/``, ``pyfresean/`` pkg)."""
    return Path(__file__).resolve().parents[3]


def fresean_pytutorial_ref_root() -> Path:
    """Root directory for stored Python tutorial FRESEAN reference outputs."""
    return Path(__file__).resolve().parent.parent / "data" / "fresean_pytutorial_ref"


def ala_dipeptide_gas_50K_reference_npz() -> Path:
    """Stored pyfresean FRESEAN reference for alanine dipeptide (gas, 50 K)."""
    return (
        fresean_pytutorial_ref_root()
        / "ala_dipeptide_gas_50K"
        / "fresean_gas_50K_reference.npz"
    )

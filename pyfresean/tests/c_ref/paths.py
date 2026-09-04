"""Paths for pyfresean vs FRESEAN COARSE (C) comparison tests."""

from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    """``pyfresean/`` project directory (contains ``examples/``, ``pyfresean/`` pkg)."""
    return Path(__file__).resolve().parents[3]


def fresean_c_ref_root() -> Path:
    """Root directory for stored FRESEAN COARSE (C) reference outputs."""
    return Path(__file__).resolve().parent.parent / "data" / "fresean_c_ref"


def ala_dipeptide_gas_300K_reference_dir() -> Path:
    """Stored C reference for alanine dipeptide (gas, 300 K)."""
    return fresean_c_ref_root() / "ala_dipeptide_gas_300K"


def hewl_solution_303K_reference_dir() -> Path:
    """Stored C reference for HEWL in solution (303 K)."""
    return fresean_c_ref_root() / "hewl_solution_303K"


def ala_dipeptide_input_root() -> Path:
    """
    Input trajectories for alanine dipeptide c_ref tests.

    Defaults to ``examples/input_data``. ``PYFRESEAN_TEST_DATA`` may override but
    must lie inside the pyfresean project tree.
    """
    project = _project_root()
    env = os.environ.get("PYFRESEAN_TEST_DATA")
    if env:
        root = Path(env).expanduser().resolve()
        if project not in root.parents and root != project:
            raise ValueError(
                f"PYFRESEAN_TEST_DATA must be inside {project}, got {root}"
            )
        return root
    return project / "examples" / "input_data"

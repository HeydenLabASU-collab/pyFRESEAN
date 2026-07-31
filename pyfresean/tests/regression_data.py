"""Trajectory paths and downloads for regression tests."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

REFERENCE_NPZ = Path(__file__).resolve().parent / "data" / "fresean_gas_50K_reference.npz"

GAS_50K_TOPOL_URL = (
    "https://www.dropbox.com/scl/fi/dz5q0x8cxm268t5hveua3/topol.tpr?"
    "rlkey=3bjvvem9xvmraa2xj9lihbbqk&dl=1"
)
GAS_50K_TRAJ_URL = (
    "https://www.dropbox.com/scl/fi/0gwbhaocef8fs0sizagab/traj.trr?"
    "rlkey=di38cgr9ozsvi8c8vtsy3kp69&dl=1"
)


def tutorial_data_root() -> Path:
    """Find example input data or the download cache."""
    env = os.environ.get("PYFRESEAN_TEST_DATA")
    if env:
        return Path(env).expanduser().resolve()
    nested = Path(__file__).resolve().parents[2] / "examples" / "input_data"
    if nested.is_dir() and (nested / "harmonic-normal-modes" / "min.xyz").exists():
        return nested
    legacy = Path(__file__).resolve().parents[3] / "data"
    if legacy.is_dir():
        return legacy
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pyfresean-test-data"
    return cache


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading regression data: {path.name} ...")
        urllib.request.urlretrieve(url, path)


def ensure_gas_50k_data(data_root: Path | None = None) -> Path:
    """Return the gas-50K data root, downloading trajectories if they are missing."""
    root = (data_root or tutorial_data_root()).resolve()
    topol = root / "MD-gas-50K" / "topol.tpr"
    traj = root / "MD-gas-50K" / "traj.trr"
    ref = root / "harmonic-normal-modes" / "min.xyz"

    if not ref.exists():
        nested_ref = Path(__file__).resolve().parents[2] / "examples" / "input_data" / "harmonic-normal-modes" / "min.xyz"
        if not nested_ref.exists():
            nested_ref = Path(__file__).resolve().parents[3] / "data" / "harmonic-normal-modes" / "min.xyz"
        if nested_ref.exists():
            root = nested_ref.parents[1]
            ref = nested_ref
            topol = root / "MD-gas-50K" / "topol.tpr"
            traj = root / "MD-gas-50K" / "traj.trr"

    _download(GAS_50K_TOPOL_URL, topol)
    _download(GAS_50K_TRAJ_URL, traj)
    if not ref.exists():
        raise FileNotFoundError(
            f"harmonic-normal-modes/min.xyz not found under {root}. "
            "Set PYFRESEAN_TEST_DATA or add min.xyz under examples/input_data/harmonic-normal-modes/"
        )
    return root

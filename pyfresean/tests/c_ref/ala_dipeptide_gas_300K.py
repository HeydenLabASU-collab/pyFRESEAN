"""Alanine dipeptide (gas, 300 K) comparison helpers for pyfresean vs FRESEAN COARSE."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import MDAnalysis as mda
import numpy as np

from pyfresean import Align, CoarseGrain, FRESEAN
from pyfresean.postprocess.spectral import total_vdos
from pyfresean.tests.c_ref.io import (
    load_c_eigenvalues_dat,
    load_c_evec_mmat,
    max_abs_eigenvector_correlation,
)
from pyfresean.tests.c_ref.paths import (
    _project_root,
    ala_dipeptide_gas_300K_reference_dir,
    ala_dipeptide_input_root,
)

# Notebook ``04_CG-alanine-dipeptide-gas-300K`` / C ``covar.inp`` matched settings.
N_FRAMES_DEFAULT = 5000
DT_PS = 0.004
N_CORR = 100
SIGMA_CM1 = 10.0
N_CONSTRAINTS_AA = 6


def default_ala_dipeptide_gas_300K_script_dir() -> Path:
    """Directory with C pipeline scripts and ``topol_prot.top``."""
    return (
        _project_root()
        / "devtools"
        / "scripts"
        / "fresean_c_ref"
        / "ala_dipeptide_gas_300K"
    )


@dataclass(frozen=True)
class AlaDipeptideGas300KPaths:
    """Input trajectories and stored C reference files for comparison."""

    data_root: Path
    aa_dir: Path
    harmonic_ref: Path
    c_reference_dir: Path
    script_dir: Path

    @property
    def aa_topol(self) -> Path:
        return self.aa_dir / "topol.tpr"

    @property
    def aa_traj(self) -> Path:
        return self.aa_dir / "traj.trr"

    @property
    def topol_prot(self) -> Path:
        return self.script_dir / "topol_prot.top"

    @property
    def c_cg_traj(self) -> Path:
        return self.c_reference_dir / "traj-cg.trr"

    @property
    def c_eval_dat(self) -> Path:
        return self.c_reference_dir / "eval_covar_cg.mmat.dat"

    @property
    def c_evec_mmat(self) -> Path:
        return self.c_reference_dir / "evec_covar_cg.mmat"


def resolve_ala_dipeptide_gas_300K_paths(
    data_root: Optional[Path] = None,
    c_reference_dir: Optional[Path] = None,
) -> AlaDipeptideGas300KPaths:
    root = (data_root or ala_dipeptide_input_root()).resolve()
    script_dir = default_ala_dipeptide_gas_300K_script_dir()
    c_dir = (
        Path(c_reference_dir).resolve()
        if c_reference_dir is not None
        else Path(
            os.environ.get("PYFRESEAN_C_REF_DIR", ala_dipeptide_gas_300K_reference_dir())
        ).resolve()
    )
    return AlaDipeptideGas300KPaths(
        data_root=root,
        aa_dir=root / "MD-gas-300K",
        harmonic_ref=root / "harmonic-normal-modes" / "min.xyz",
        c_reference_dir=c_dir,
        script_dir=script_dir,
    )


def run_pyfresean_coarse_grain(
    paths: AlaDipeptideGas300KPaths,
    n_frames: int = N_FRAMES_DEFAULT,
    select: str = "all",
) -> tuple[CoarseGrain, mda.Universe]:
    if not paths.aa_topol.exists() or not paths.aa_traj.exists():
        raise FileNotFoundError(
            f"missing all-atom data under {paths.aa_dir} "
            "(need topol.tpr and traj.trr)"
        )
    cg, u_cg = CoarseGrain.cg_universe(
        (paths.aa_topol, paths.aa_traj),
        select=select,
        stop=n_frames,
    )
    return cg, u_cg


def compare_cg_trajectory_to_c_reference(
    u_cg: mda.Universe,
    reference_trr: Path,
    max_rmsd_nm: float = 0.05,
) -> dict[str, float]:
    """Compare pyfresean CG positions/velocities to C ``traj-cg.trr``."""
    ref = mda.Universe(str(reference_trr))
    if len(ref.trajectory) < len(u_cg.trajectory):
        raise ValueError("reference trajectory has fewer frames than pyfresean CG")
    n_frames = len(u_cg.trajectory)
    pos_diff = []
    vel_diff = []
    for frame_index in range(n_frames):
        u_cg.trajectory[frame_index]
        ref.trajectory[frame_index]
        pos_diff.append(
            np.linalg.norm(u_cg.atoms.positions - ref.atoms.positions)
        )
        if u_cg.atoms.velocities is None or ref.atoms.velocities is None:
            continue
        vel_diff.append(
            np.linalg.norm(u_cg.atoms.velocities - ref.atoms.velocities)
        )
    max_pos = float(np.max(pos_diff))
    mean_pos = float(np.mean(pos_diff))
    metrics = {
        "max_position_norm_angstrom": max_pos,
        "mean_position_norm_angstrom": mean_pos,
        "max_position_rmsd_nm": max_pos / np.sqrt(u_cg.atoms.n_atoms) / 10.0,
    }
    if vel_diff:
        max_vel = float(np.max(vel_diff))
        metrics["max_velocity_norm"] = max_vel
        metrics["mean_velocity_norm"] = float(np.mean(vel_diff))
    if metrics["max_position_rmsd_nm"] > max_rmsd_nm:
        raise AssertionError(
            f"CG position mismatch: max RMSD {metrics['max_position_rmsd_nm']:.4f} nm "
            f"> {max_rmsd_nm} nm"
        )
    return metrics


def run_pyfresean_fresean_cg(
    cg: CoarseGrain,
    u_cg: mda.Universe,
    paths: AlaDipeptideGas300KPaths,
    lag_symmetrization: str = "average",
) -> FRESEAN:
    if not paths.harmonic_ref.exists():
        raise FileNotFoundError(f"missing reference structure {paths.harmonic_ref}")
    u_ref = mda.Universe(str(paths.harmonic_ref))
    ref_cg = cg.map_positions(u_ref.atoms.positions)
    u_cg.trajectory.add_transformations(
        Align(u_cg.atoms, reference_positions=ref_cg, place_com_in_box=False),
    )
    analysis = FRESEAN(
        u_cg,
        select="all",
        n_constraints=cg.mapping.n_constraints,
        n_corr=N_CORR,
        dt=DT_PS,
        sigma=SIGMA_CM1,
        lag_symmetrization=lag_symmetrization,
    )
    analysis.run()
    return analysis


def _c_eigenvalues_vdos_normalized(
    c_eval: np.ndarray,
    n_corr: int,
    win_time_0: float,
    n_dof: int,
) -> np.ndarray:
    """Apply the same VDOS normalization pyfresean uses after diagonalization."""
    avg_temp = (
        np.sum(c_eval[0])
        + 2 * np.sum(c_eval[1:])
    ) / (2 * n_corr - 1) / win_time_0 / (8.3145 * 0.1) / n_dof
    vdos_norm = n_corr * win_time_0 * (8.3145 * 0.1 * avg_temp)
    if vdos_norm <= 0:
        return c_eval
    return c_eval / vdos_norm


def compare_fresean_eigenvalues_to_c(
    analysis: FRESEAN,
    eval_dat: Path,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    n_freq: int | None = None,
) -> dict[str, float]:
    c_eval = _c_eigenvalues_vdos_normalized(
        load_c_eigenvalues_dat(eval_dat),
        analysis.n_corr,
        analysis.results.win_time[0],
        analysis._n_dof,
    )
    py_eval = analysis.results.eigenvalues
    n_compare = min(
        c_eval.shape[0],
        py_eval.shape[0],
        n_freq if n_freq is not None else py_eval.shape[0],
    )
    n_dof = min(c_eval.shape[1], py_eval.shape[1])
    np.testing.assert_allclose(
        py_eval[:n_compare, :n_dof],
        c_eval[:n_compare, :n_dof],
        rtol=rtol,
        atol=atol,
    )
    return {
        "n_freq_compared": float(n_compare),
        "n_dof_compared": float(n_dof),
        "max_abs_diff": float(
            np.max(np.abs(py_eval[:n_compare, :n_dof] - c_eval[:n_compare, :n_dof]))
        ),
    }


def load_c_vdos_total_normalized(analysis: FRESEAN, eval_dat: Path) -> np.ndarray:
    """Total VDoS from C eigenvalues with the same normalization as pyfresean."""
    c_eval = _c_eigenvalues_vdos_normalized(
        load_c_eigenvalues_dat(eval_dat),
        analysis.n_corr,
        analysis.results.win_time[0],
        analysis._n_dof,
    )
    return total_vdos(c_eval)


def compare_fresean_vdos_total_to_c(
    analysis: FRESEAN,
    eval_dat: Path,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> dict[str, float]:
    """Compare full total VDoS curves (all frequency bins) to C ``eval`` reference."""
    c_vdos = load_c_vdos_total_normalized(analysis, eval_dat)
    py_vdos = analysis.results.vdos_total
    n_compare = min(c_vdos.shape[0], py_vdos.shape[0])
    np.testing.assert_allclose(
        py_vdos[:n_compare],
        c_vdos[:n_compare],
        rtol=rtol,
        atol=atol,
    )
    return {
        "n_freq_compared": float(n_compare),
        "max_abs_diff": float(np.max(np.abs(py_vdos[:n_compare] - c_vdos[:n_compare]))),
        "py_vdos_sum": float(py_vdos[:n_compare].sum()),
        "c_vdos_sum": float(c_vdos[:n_compare].sum()),
    }


def compare_fresean_eigenvectors_to_c(
    analysis: FRESEAN,
    evec_mmat: Path,
    freq_indices: tuple[int, ...] = (0, 1),
    n_modes: int | None = 12,
    min_correlation: float = 0.99,
) -> dict[str, float]:
    c_evec = load_c_evec_mmat(evec_mmat, n_corr=analysis.n_corr)
    py_evec = analysis.results.eigenvectors
    n_compare = min(
        py_evec.shape[1],
        c_evec.shape[1],
        n_modes if n_modes is not None else py_evec.shape[1],
    )
    metrics = {}
    for freq_index in freq_indices:
        corr = max_abs_eigenvector_correlation(
            py_evec,
            c_evec,
            freq_index,
            n_compare,
        )
        metrics[f"max_mode_correlation_freq_{freq_index}"] = corr
        if corr < min_correlation:
            raise AssertionError(
                f"eigenvector correlation at freq {freq_index} is {corr:.4f} "
                f"< {min_correlation}"
            )
    return metrics

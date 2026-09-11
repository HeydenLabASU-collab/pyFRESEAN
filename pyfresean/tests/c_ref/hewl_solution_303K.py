"""HEWL in solution (303 K) comparison helpers for pyfresean vs FRESEAN COARSE."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import MDAnalysis as mda

from pyfresean import Align, CoarseGrain, FRESEAN
from pyfresean.tests.c_ref.ala_dipeptide_gas_300K import (
    N_CORR,
    SIGMA_CM1,
    compare_cg_trajectory_to_c_reference,
    compare_fresean_eigenvalues_to_c,
    compare_fresean_eigenvectors_to_c,
    compare_fresean_vdos_total_to_c,
    load_c_vdos_total_normalized,
)
from pyfresean.tests.c_ref.paths import (
    _project_root,
    hewl_solution_303K_reference_dir,
)

N_FRAMES_DEFAULT = 5000
DT_PS = 0.01  # input.mdp: dt=2 fs, nstxout=5
# HEWL has ~250 CG beads (738 DOF); plot/compare all modes (ala dipeptide uses 12).
N_MODES_COMPARE = None


def default_hewl_solution_303K_script_dir() -> Path:
    return (
        _project_root()
        / "devtools"
        / "scripts"
        / "fresean_c_ref"
        / "hewl_solution_303K"
    )


@dataclass(frozen=True)
class HewlSolution303KPaths:
    """Protein trajectory and stored C reference files."""

    data_dir: Path
    c_reference_dir: Path
    script_dir: Path

    @property
    def aa_topol(self) -> Path:
        return self.data_dir / "output" / "topol_prot.tpr"

    @property
    def aa_traj(self) -> Path:
        return self.data_dir / "output" / "sample-NPT_prot_pbc.trr"

    @property
    def topol_prot(self) -> Path:
        return self.data_dir / "input" / "topol_prot.top"

    @property
    def c_cg_traj(self) -> Path:
        return self.c_reference_dir / "traj-cg.trr"

    @property
    def c_eval_dat(self) -> Path:
        return self.c_reference_dir / "eval_covar_cg.mmat.dat"

    @property
    def c_evec_mmat(self) -> Path:
        return self.c_reference_dir / "evec_covar_cg.mmat"


def resolve_hewl_solution_303K_paths(
    data_dir: Optional[Path] = None,
    c_reference_dir: Optional[Path] = None,
) -> HewlSolution303KPaths:
    data = (
        Path(data_dir).resolve()
        if data_dir is not None
        else Path(
            os.environ.get(
                "PYFRESEAN_HEWL_DATA_DIR", hewl_solution_303K_reference_dir()
            )
        ).resolve()
    )
    c_dir = (
        Path(c_reference_dir).resolve()
        if c_reference_dir is not None
        else Path(os.environ.get("PYFRESEAN_C_REF_DIR", data)).resolve()
    )
    return HewlSolution303KPaths(
        data_dir=data,
        c_reference_dir=c_dir,
        script_dir=default_hewl_solution_303K_script_dir(),
    )


def run_pyfresean_coarse_grain(
    paths: HewlSolution303KPaths,
    n_frames: int = N_FRAMES_DEFAULT,
    select: str = "all",
):
    if not paths.aa_topol.exists() or not paths.aa_traj.exists():
        raise FileNotFoundError(
            f"missing protein trajectory under {paths.data_dir}/output "
            "(need topol_prot.tpr and sample-NPT_prot_pbc.trr)"
        )
    return CoarseGrain.cg_universe(
        (paths.aa_topol, paths.aa_traj),
        select=select,
        stop=n_frames,
    )


def run_pyfresean_fresean_cg(
    cg: CoarseGrain,
    u_cg: mda.Universe,
    lag_symmetrization: str = "average",
) -> FRESEAN:
    u_cg.trajectory[0]
    ref_cg = u_cg.atoms.positions.copy()
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


__all__ = [
    "DT_PS",
    "HewlSolution303KPaths",
    "N_FRAMES_DEFAULT",
    "N_MODES_COMPARE",
    "compare_cg_trajectory_to_c_reference",
    "compare_fresean_eigenvalues_to_c",
    "compare_fresean_eigenvectors_to_c",
    "compare_fresean_vdos_total_to_c",
    "load_c_vdos_total_normalized",
    "resolve_hewl_solution_303K_paths",
    "run_pyfresean_coarse_grain",
    "run_pyfresean_fresean_cg",
]

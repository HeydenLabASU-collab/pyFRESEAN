"""Regenerate ``fresean_gas_50K_reference.npz`` for ala dipeptide (gas, 50 K) pytutorial_ref tests.

Same FRESEAN setup as ``examples/01_MD-alanine-dipeptide-gas-50K.ipynb``.

Usage::

    python devtools/scripts/generate_fresean_pytutorial_ref_ala_dipeptide_gas_50K.py \\
        --data-dir examples/input_data
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import MDAnalysis as mda
import numpy as np

from pyfresean import Align, FRESEAN
from pyfresean.postprocess import low_frequency_peaks

FRESEAN_TUTORIAL_URL = "https://github.com/HeydenLabASU-collab/FRESEAN_tutorial"
EXAMPLE_NOTEBOOK = "examples/01_MD-alanine-dipeptide-gas-50K.ipynb"


def default_data_dir() -> Path:
    env = os.environ.get("PYFRESEAN_TEST_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "examples" / "input_data"


def default_output_npz() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "pyfresean"
        / "tests"
        / "data"
        / "fresean_pytutorial_ref"
        / "ala_dipeptide_gas_50K"
        / "fresean_gas_50K_reference.npz"
    )


def run_fresean_gas_50k(data_dir: Path) -> dict:
    topol = data_dir / "MD-gas-50K" / "topol.tpr"
    traj = data_dir / "MD-gas-50K" / "traj.trr"
    ref = data_dir / "harmonic-normal-modes" / "min.xyz"
    for path in (topol, traj, ref):
        if not path.exists():
            raise FileNotFoundError(path)

    n_corr, dt, sigma, n_constraints = 500, 0.004, 10.0, 6

    u = mda.Universe(str(topol), str(traj))
    sel = u.select_atoms("all")
    u_ref = mda.Universe(str(ref))
    u.trajectory.add_transformations(
        Align(sel, reference_positions=u_ref.atoms.positions, place_com_in_box=False),
    )

    analysis = FRESEAN(
        u,
        select="all",
        n_constraints=n_constraints,
        n_corr=n_corr,
        dt=dt,
        sigma=sigma,
    )
    analysis.run()

    freqs = analysis.results.freqs
    vdos = analysis.results.vdos_total
    low_peaks = low_frequency_peaks(freqs, vdos, max_freq=200.0)

    return {
        "freqs": freqs,
        "vdos_total": vdos,
        "avg_temperature": analysis.results.avg_temperature,
        "n_dof": analysis.results.n_dof,
        "vdos_sum": vdos.sum(),
        "n_frames": analysis.n_frames,
        "n_corr": n_corr,
        "dt": dt,
        "sigma": sigma,
        "n_constraints": n_constraints,
        "low_peak_indices": low_peaks,
        "low_peak_freqs": freqs[low_peaks],
        "eigenvalues_f0": analysis.results.eigenvalues[0],
        "eigenvalues_f1": analysis.results.eigenvalues[1],
        "source": f"{EXAMPLE_NOTEBOOK} ({FRESEAN_TUTORIAL_URL})",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Directory with MD-gas-50K/ and harmonic-normal-modes/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_npz(),
        help="Output .npz path",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    results = run_fresean_gas_50k(data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **results)
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
    print(f"T = {results['avg_temperature']:.2f} K")
    print(f"peaks (cm-1) = {results['low_peak_freqs']}")


if __name__ == "__main__":
    main()

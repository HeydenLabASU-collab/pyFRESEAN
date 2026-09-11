"""Regenerate stored FRESEAN COARSE (C) reference for ala dipeptide (gas, 300 K) c_ref tests.

Writes to ``pyfresean/tests/data/fresean_c_ref/ala_dipeptide_gas_300K/``.

Requires ``fresean`` (FRESEANCOARSE), ``gmx``, and GSL for ``fresean eigen``.

Usage::

    module load gsl
    export PATH=/path/to/FRESEANCOARSE/bin:$PATH
    export GMX=/path/to/gmx
    python devtools/scripts/generate_fresean_c_ref_ala_dipeptide_gas_300K.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Input tree with MD-gas-300K/ (default: examples/input_data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Reference directory "
            "(default: tests/data/fresean_c_ref/ala_dipeptide_gas_300K)"
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    script = (
        project_root
        / "devtools"
        / "scripts"
        / "fresean_c_ref"
        / "ala_dipeptide_gas_300K"
        / "run_c_pipeline.sh"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root
        / "pyfresean"
        / "tests"
        / "data"
        / "fresean_c_ref"
        / "ala_dipeptide_gas_300K"
    )
    data_dir = args.data_dir or project_root / "examples" / "input_data"

    env = {
        **dict(__import__("os").environ),
        "PYFRESEAN_C_REF_DIR": str(output_dir),
    }
    if args.data_dir is not None:
        env["PYFRESEAN_TEST_DATA"] = str(data_dir.resolve())

    print(f"Input data: {data_dir}")
    print(f"Writing C reference to: {output_dir}")
    subprocess.run(["bash", str(script)], check=True, env=env)

    required = [
        "traj-cg.trr",
        "eval_covar_cg.mmat.dat",
        "evec_covar_cg.mmat",
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        print(f"Pipeline finished but missing: {missing}", file=sys.stderr)
        sys.exit(1)

    for name in required:
        path = output_dir / name
        print(f"  {name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()

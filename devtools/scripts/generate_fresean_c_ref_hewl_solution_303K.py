"""Regenerate stored FRESEAN COARSE (C) reference for HEWL (solution, 303 K) c_ref tests."""

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
        help="hewl_solution_303K directory with output/ and input/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write traj-cg.trr, eval, evec (default: data-dir)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    script = (
        project_root
        / "devtools"
        / "scripts"
        / "fresean_c_ref"
        / "hewl_solution_303K"
        / "run_c_pipeline.sh"
    )
    data_dir = (
        args.data_dir.resolve()
        if args.data_dir is not None
        else project_root
        / "pyfresean"
        / "tests"
        / "data"
        / "fresean_c_ref"
        / "hewl_solution_303K"
    )
    output_dir = (
        args.output_dir.resolve() if args.output_dir is not None else data_dir
    )

    env = {
        **dict(__import__("os").environ),
        "HEWL_DATA_DIR": str(data_dir),
        "PYFRESEAN_C_REF_DIR": str(output_dir),
    }

    print(f"HEWL data: {data_dir}")
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

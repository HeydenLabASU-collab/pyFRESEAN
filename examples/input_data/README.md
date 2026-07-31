# Input data

Trajectory and reference files for the example notebooks.

| Path | Contents |
|------|----------|
| `harmonic-normal-modes/` | `min.xyz` and harmonic mode data (gas-phase notebooks) |
| `HEWL/1hel.pdb` | HEWL starting structure (notebook 05) |
| `MD-gas-50K/`, `MD-gas-300K/`, `MD-water-300K/` | GROMACS `topol.tpr` and `traj.trr` (notebooks 01–03, 04) |

## Trajectory files

Trajectory files are not in the git repository (`*.trr` and `*.tpr` are gitignored). Only the small reference structures in the table above are committed.

To get the trajectories, either:

1. **Run the notebook** — notebooks 01–04 download missing `topol.tpr` / `traj.trr` from Dropbox URLs in the first code cell (gas 50K is about 155 MB).
2. **Copy from FRESEAN_tutorial** — clone [FRESEAN_tutorial](https://github.com/HeydenLabASU-collab/FRESEAN_tutorial) and copy the `data/` folders into this directory, keeping the same names (`MD-gas-50K/`, `MD-gas-300K/`, `MD-water-300K/`).

Notebook 05 uses HEWL: `1hel.pdb` is included here; the protein TRR from unbiased MD needs to be produced with GROMACS as indicated in the notebook (see `05_CG-hewl.ipynb`, §0).

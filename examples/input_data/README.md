# Input data

Trajectory and reference files for the example notebooks.

| Path | Contents |
|------|----------|
| `MD-alanine-dipeptide-300K/` | Gas-phase ala dipeptide `topol.tpr` / `traj.trr` / `min.xyz` (notebook 01; downloaded by the notebook if missing) |
| `MD-HEWL-303K/` | HEWL solution MD inputs and scripts (`run-slurm.sh`, `postprocess-prot.sh`, `1hel.pdb`, `input/`) |

## Trajectory files

Trajectory files are not in the git repository (`*.trr` and `*.tpr` are gitignored). Only small reference structures (e.g. `1hel.pdb`, `min.xyz` when committed) live here.

### Alanine dipeptide (notebook 01)

Run the notebook — it downloads missing `topol.tpr` / `traj.trr` from Dropbox URLs in the first code cell.

### HEWL (notebooks 02–04)

1. Run [`MD-HEWL-303K/run-slurm.sh`](MD-HEWL-303K/run-slurm.sh) to produce `MD-HEWL-303K/output/run.tpr` and `run.trr`.
2. **Notebook 02 / 03** also need protein-only, PBC-corrected files from [`MD-HEWL-303K/postprocess-prot.sh`](MD-HEWL-303K/postprocess-prot.sh) (`topol_prot.tpr`, `sample-NPT_prot_pbc.trr`).
3. **Notebook 04 (streaming)** uses only the raw `run.tpr` / `run.trr`; MDAnalysis `Unwrap` replaces the GROMACS postprocess step, then CG is materialized in memory for FRESEAN (no AA FRESEAN pass).

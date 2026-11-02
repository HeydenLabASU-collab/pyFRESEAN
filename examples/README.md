# pyFRESEAN examples

Example notebooks for FRESEAN analysis with the `pyfresean` package.

| Notebook | System | What it covers |
|---|---|---|
| `01_AA-alanine-dipeptide-gas-300K.ipynb` | Alanine dipeptide, all-atom, gas, 300 K | Align, FRESEAN, total VDoS, per-mode VDoS (modes 1 & 2) |
| `02_AA_CG-hewl-solution-300K.ipynb` | HEWL, all-atom + coarse-grained, solution, 300 K | CG mapping, FRESEAN on AA and CG, total VDoS overlay, per-mode VDoS (modes 7 & 8); uses protein-only postprocessed TRR |
| `03_CG-hewl-fresean-parallel-benchmark.ipynb` | HEWL, CG only, solution, 300 K | CPU detection, serial vs per-phase `parallel=` settings, `benchmark=True` timings, and summary table |
| `04_CG-hewl-solution-streaming.ipynb` | HEWL, CG only, solution, 300 K | Stream raw solvated `run.tpr`/`run.trr` with MDAnalysis `Unwrap`, build in-memory CG, FRESEAN on CG (no postprocess / no AA FRESEAN) |

Run notebooks from the `examples/` directory so `input_data/` and `output_data/` resolve correctly.

- **Inputs:** `input_data/` — see `input_data/README.md` for trajectory files.
- **Outputs:** `output_data/` — generated files; gitignored except the README.

[`01_AA-alanine-dipeptide-gas-300K.ipynb`](01_AA-alanine-dipeptide-gas-300K.ipynb) downloads missing alanine dipeptide trajectories automatically.
[`02_AA_CG-hewl-solution-300K.ipynb`](02_AA_CG-hewl-solution-300K.ipynb) expects HEWL **protein-only** files from GROMACS MD + [`postprocess-prot.sh`](input_data/MD-HEWL-303K/postprocess-prot.sh) (see the notebook and [`input_data/README.md`](input_data/README.md)).
[`03_CG-hewl-fresean-parallel-benchmark.ipynb`](03_CG-hewl-fresean-parallel-benchmark.ipynb) expects the same HEWL protein trajectory files as notebook 02.
[`04_CG-hewl-solution-streaming.ipynb`](04_CG-hewl-solution-streaming.ipynb) expects only the raw solvated outputs from [`run-slurm.sh`](input_data/MD-HEWL-303K/run-slurm.sh) (`output/run.tpr`, `output/run.trr`); PBC unwrap is done in MDAnalysis, and FRESEAN runs on the in-memory CG universe only.
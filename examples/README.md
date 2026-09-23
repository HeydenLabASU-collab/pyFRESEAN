# pyFRESEAN examples

Three example notebooks for FRESEAN analysis with the `pyfresean` package.

| Notebook | System | What it covers |
|---|---|---|
| `01_AA-alanine-dipeptide-gas-300K.ipynb` | Alanine dipeptide, all-atom, gas, 300 K | Align, FRESEAN, total VDoS, per-mode VDoS (modes 1 & 2) |
| `02_AA_CG-hewl-solution-300K.ipynb` | HEWL, all-atom + coarse-grained, solution, 300 K | CG mapping, FRESEAN on AA and CG, total VDoS overlay, per-mode VDoS (modes 7 & 8) |
| `03_CG-hewl-fresean-parallel-benchmark.ipynb` | HEWL, CG only, solution, 300 K | CPU detection, serial vs per-phase `parallel=` settings, `benchmark=True` timings, and summary table |

Run notebooks from the `examples/` directory so `input_data/` and `output_data/` resolve correctly.

- **Inputs:** `input_data/` — see `input_data/README.md` for trajectory files.
- **Outputs:** `output_data/` — generated files; gitignored except the README.

[`01_AA-alanine-dipeptide-gas-300K.ipynb`](01_AA-alanine-dipeptide-gas-300K.ipynb) downloads missing alanine dipeptide trajectories automatically. 
[`02_AA_CG-hewl-solution-300K.ipynb`](02_AA_CG-hewl-solution-300K.ipynb) expects HEWL protein trajectory files from a GROMACS run (see the notebook and [`input_data/README.md`](input_data/README.md)).
[`03_CG-hewl-fresean-parallel-benchmark.ipynb`](03_CG-hewl-fresean-parallel-benchmark.ipynb) expects the same HEWL protein trajectory files as in [`02_AA_CG-hewl-solution-300K.ipynb`](02_AA_CG-hewl-solution-300K.ipynb).
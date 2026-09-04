# HEWL in solution (303 K) — GROMACS + FRESEAN c_ref

Input from [imd-workshop-2025](https://github.com/amruthesht/imd-workshop-2025/tree/main/workshop/sample_simulation/GROMACS/input)
(lysozyme, CHARMM-GUI, ~30k atoms equilibrated `start.gro`).

## GROMACS (0.1 ns production)

- `input/input.mdp`: TRR with velocities (`nstxout = nstvout = 5`, `nsteps = 50000`)
- `sbatch run-slurm.sh` → `output/run.trr`
- `bash postprocess-prot.sh` → `output/topol_prot.tpr`, `output/sample-NPT_prot_pbc.trr`

## FRESEAN COARSE (C) reference

| File | C step |
|------|--------|
| `traj-cg.trr` | `fresean coarse` |
| `eval_covar_cg.mmat.dat` | `fresean eigen` |
| `evec_covar_cg.mmat` | `fresean eigen` |

Parameters: `n_corr=100`, `dt=0.01` ps, `sigma=10` cm⁻¹, `5000` frames, alignment to first CG frame.

Regenerate C reference:

```bash
module load gsl gromacs
export PATH=/path/to/FRESEANCOARSE/bin:$PATH
cd pyfresean
bash devtools/scripts/fresean_c_ref/hewl_solution_303K/generate_topol_prot.sh
python devtools/scripts/generate_fresean_c_ref_hewl_solution_303K.py
```

## Tests (optional, `@pytest.mark.c_ref`)

```bash
pytest -m c_ref pyfresean/tests/test_fresean_regression_hewl_solution_303K_c_ref.py
pytest -m c_ref --c-ref-plot pyfresean/tests/test_fresean_regression_hewl_solution_303K_c_ref.py
```

Plots: `plots/eigenvalues_pyfresean_vs_c.png`, `plots/vdos_total_pyfresean_vs_c.png` (gitignored; generate with `--c-ref-plot`)

## Git tracking

Committed (like ala dipeptide `c_ref`):

- GROMACS inputs: `input/` (`input.mdp`, `start.gro`, `topol.top`, `index.ndx`, `toppar/`, `topol_prot.top`)
- Scripts: `run-slurm.sh`, `postprocess-prot.sh`, `README.md`
- C reference: `traj-cg.trr`, `eval_covar_cg.mmat.dat`

Not committed (regenerate locally):

- `output/` — full MD trajectories (`run.trr` is multi-GB)
- `evec_covar_cg.mmat` — ~400 MB; run `generate_fresean_c_ref_hewl_solution_303K.py`
- `plots/`, SLURM logs, pipeline staging files

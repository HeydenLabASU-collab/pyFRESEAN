# FRESEAN COARSE (C) reference — alanine dipeptide (gas, 300 K)

Stored outputs from **FRESEAN COARSE** for ``c_ref`` pytest checks.

| File | C step |
|------|--------|
| `traj-cg.trr` | `fresean coarse` |
| `eval_covar_cg.mmat.dat` | `fresean eigen` (eigenvalues) |
| `evec_covar_cg.mmat` | `fresean eigen` (eigenvectors) |

Parameters match ``examples/04_CG-alanine-dipeptide-gas-300K.ipynb``:
`n_corr=100`, `dt=0.004` ps, `sigma=10` cm⁻¹, `5000` frames.

## Regenerate

```bash
module load gsl
export PATH=/path/to/FRESEANCOARSE/bin:$PATH
export GMX=/path/to/gmx

cd pyfresean
python devtools/scripts/generate_fresean_c_ref_ala_dipeptide_gas_300K.py
```

## Run tests

```bash
pytest -m c_ref pyfresean/tests/test_fresean_regression_ala_dipeptide_gas_300K_c_ref.py
```

Optional comparison plots (pyfresean vs C, PNG):

```bash
pytest -m c_ref --c-ref-plot \
  pyfresean/tests/test_fresean_regression_ala_dipeptide_gas_300K_c_ref.py

# custom output directory:
pytest -m c_ref --c-ref-plot --c-ref-plot-dir /path/to/plots \
  pyfresean/tests/test_fresean_regression_ala_dipeptide_gas_300K_c_ref.py
```

Default plot paths:
- ``tests/data/fresean_c_ref/ala_dipeptide_gas_300K/plots/eigenvalues_pyfresean_vs_c.png``
- ``tests/data/fresean_c_ref/ala_dipeptide_gas_300K/plots/vdos_total_pyfresean_vs_c.png``

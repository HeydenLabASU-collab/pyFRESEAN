# Regression reference data

## `fresean_gas_50K_reference.npz`

Stored FRESEAN outputs for alanine dipeptide (gas, 50 K), using the same setup as `examples/01_MD-alanine-dipeptide-gas-50K.ipynb` and the Python FRESEAN notebooks in [FRESEAN_tutorial](https://github.com/HeydenLabASU-collab/FRESEAN_tutorial).

Parameters: align to `harmonic-normal-modes/min.xyz`, `n_corr=500`, `dt=0.004` ps, `sigma=10` cm⁻¹, `n_constraints=6`, full trajectory (~250k frames).

### Regenerate the reference file

After changes to the FRESEAN core, rebuild with:

```bash
python devtools/scripts/generate_fresean_regression_reference.py \
  --data-dir examples/input_data
```

### Run regression tests

Regression tests are skipped by default. To run them:

```bash
pytest -m regression
```

Trajectory data: use `examples/input_data`, set `PYFRESEAN_TEST_DATA` to a local `data/` tree, or let the tests download to the cache (~155 MB on first run).

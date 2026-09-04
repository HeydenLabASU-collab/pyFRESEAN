# Regression reference — alanine dipeptide (gas, 50 K)

## `fresean_gas_50K_reference.npz`

Stored FRESEAN outputs for alanine dipeptide (gas, 50 K), using the same setup as
`examples/01_MD-alanine-dipeptide-gas-50K.ipynb`.

Parameters: align to `harmonic-normal-modes/min.xyz`, `n_corr=500`, `dt=0.004` ps,
`sigma=10` cm⁻¹, `n_constraints=6`, full trajectory (~250k frames).

### Regenerate

```bash
python devtools/scripts/generate_fresean_pytutorial_ref_ala_dipeptide_gas_50K.py \
  --data-dir examples/input_data
```

### Run tests

```bash
pytest -m pytutorial_ref pyfresean/tests/test_fresean_regression_ala_dipeptide_gas_50K_pytutorial_ref.py
```

Trajectory data: `examples/input_data`, `PYFRESEAN_TEST_DATA`, or test download cache.

# HEWL benchmark results

Spectral sweep outputs only. One-time CG lives in `../cg_reference/`.

```
results/
├── c_spectral/     # --mode c_spectral
├── py_cases/       # --mode py_fresean
└── plots/          # collect_results.py --plot
```

CG timings: `../cg_reference/pyfresean/result.json`, `../cg_reference/c_ref/result.json`

```bash
python bench/hewl/collect_results.py --case all --plot
```

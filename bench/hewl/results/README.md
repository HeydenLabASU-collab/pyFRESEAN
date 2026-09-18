# HEWL benchmark results

Spectral sweep outputs. One-time reference inputs live in `../cg_reference/` and `../aa_reference/`.

```
results/
├── results_cg/
│   ├── c_spectral/
│   ├── py_cases/
│   └── plots/
└── results_aa/
    ├── c_spectral/
    ├── py_cases/
    └── plots/
```

```bash
python bench/hewl/collect_results.py --system cg --case all --plot
python bench/hewl/collect_results.py --system aa --case omp_n_njobs_1 --plot
```

Console tables include speedup and \(1/T\). With `--plot`, see also
`speedup_all_cases.png`, `inverse_walltime_all_cases.png`, and
`vectorized_corr_comparison.png` when vectorized results exist.

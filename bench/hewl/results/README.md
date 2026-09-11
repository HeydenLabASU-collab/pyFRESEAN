# HEWL benchmark results

Timed sweeps only. Shared CG inputs live in `bench/hewl/cg_reference/`.

```
results/
├── c_spectral/            # C covar+eigen per ncpus
├── py_cases/              # py FRESEAN per case × ncpus
│   ├── omp1_njobs_n_nworkers_1/
│   ├── omp1_njobs_n_nworkers_2/
│   ├── omp1_njobs_n_nworkers_n/
│   └── omp_n_njobs_1/
└── plots/                 # from collect_results.py --plot
```

CG reference timings and cache: `../cg_reference/pyfresean/result.json`

## Collect / plot

```bash
python bench/hewl/collect_results.py --case all --plot
```

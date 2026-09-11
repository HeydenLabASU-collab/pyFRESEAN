# HEWL pyfresean benchmark

Wall-clock FRESEAN spectral analysis on HEWL (303 K, 5000 frames) vs FRESEAN COARSE (C).

## Layout

```
bench/hewl/
├── benchmark.py       # py_cg, c_cg, py_fresean, c_spectral, all
├── collect_results.py
├── submit.sh
├── cg_reference/      # one-time CG (gitignored except README)
│   ├── c_ref/
│   └── pyfresean/
├── aa_reference/      # one-time AA inputs for C (gitignored except README)
│   └── c_ref/
└── results/           # spectral sweeps (gitignored except README)
    ├── results_cg/
    │   ├── c_spectral/
    │   ├── py_cases/
    │   └── plots/
    └── results_aa/
        ├── c_spectral/
        ├── py_cases/
        └── plots/
```

## Modes

| Mode | What | Output |
|------|------|--------|
| `py_cg` | py coarse-grain once | `cg_reference/pyfresean/` |
| `c_cg` | C `fresean coarse` once | `cg_reference/c_ref/` |
| `c_aa` | C all-atom inputs once | `aa_reference/c_ref/` |
| `py_fresean` | py spectral sweep | `results/results_{cg,aa}/py_cases/{case}/` |
| `c_spectral` | C covar+eigen sweep | `results/results_{cg,aa}/c_spectral/` |

Plots compare `py_fresean` vs `c_spectral` only (CG excluded from both sides).

## Prerequisites

- HEWL data: `pyfresean/tests/data/fresean_c_ref/hewl_solution_303K/`
- `FRESEAN_BIN`, `EIGEN_BIN` on PATH; `module load gsl gromacs`
- `pip install -e .` from pyfresean root

## Run

One-time CG (or use `submit.sh all`):

```bash
python bench/hewl/benchmark.py --mode py_cg --ncpus 1
python bench/hewl/benchmark.py --mode c_cg --ncpus 1
```

Full sweep:

```bash
bash bench/hewl/submit.sh all
```

Plot:

```bash
python bench/hewl/collect_results.py --system cg --case all --plot
python bench/hewl/collect_results.py --system aa --case omp_n_njobs_1 --plot
```

## `py_fresean` cases

| Case | OMP | n_jobs | n_workers |
|------|-----|--------|-----------|
| omp1_njobs_n_nworkers_1 | 1 | N | 1 |
| omp1_njobs_n_nworkers_2 | 1 | N | min(2, N) |
| omp1_njobs_n_nworkers_n | 1 | N | N |
| omp_n_njobs_1 | N | 1 | 1 |

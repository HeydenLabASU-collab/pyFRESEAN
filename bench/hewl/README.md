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

Each case maps to a ``FRESEAN(parallel=...)`` dict via
``fresean_parallel_for_case(case, ncpus)`` in ``benchmark.py``.

| Case | velocity_fft | corr_matrix | eigen |
|------|--------------|-------------|-------|
| omp_n_njobs_1_vec | omp=N | omp=N | omp=N |
| omp1_njobs_n_vec | omp=1 | n_jobs=N, omp=1 | omp=1 |
| omp_hybrid_vec | omp=N | n_jobs=N, omp=1 | omp=N |

Submit:

```bash
bash bench/hewl/submit.sh py_fresean_vec     # uniform + tile-pool vec
bash bench/hewl/submit.sh py_fresean_hybrid  # hybrid only
```

## Plots (`collect_results.py --plot`)

Per case under `results/results_{cg,aa}/plots/{case}/`:

- `total_vs_c.png` — wall time vs C covar+eigen
- `breakdown.png` — velocity / corr matrix / eigen phases
- `speedup.png` — \(T_1 / T_N\)
- `inverse_walltime.png` — \(1 / T_N\)

Overview under `plots/`:

- `total_all_cases.png`
- `inverse_walltime_all_cases.png`
- `speedup_all_cases.png`
- `vectorized_corr_comparison.png` — `omp_n_njobs_1` vs `omp_n_njobs_1_vec`

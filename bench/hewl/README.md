# HEWL pyfresean benchmark

Wall-clock FRESEAN spectral analysis on HEWL (303 K, 5000 frames) vs FRESEAN COARSE (C).

## Layout

```
bench/hewl/
├── benchmark.py              # run one benchmark job (modes: cg, fresean, c, all)
├── collect_results.py        # summarize + plot results
├── prepare_c_covar_inputs.sh # one-time C CG inputs → cg_reference/c_ref/
├── submit.sh                 # submit SLURM sweeps (cg, c, py phases)
├── cg_reference/             # shared CG artifacts (outside results/, gitignored)
│   ├── c_ref/                # C fresean inputs: topol-cg.mtop, ref-cg.gro, traj-cg.trr
│   └── pyfresean/            # py CG cache + result.json (bench_t_py_coarse)
└── results/
    ├── c_spectral/           # C covar+eigen per ncpus
    ├── py_cases/             # py FRESEAN per case × ncpus
    │   ├── omp1_njobs_n_nworkers_1/
    │   ├── omp1_njobs_n_nworkers_2/
    │   ├── omp1_njobs_n_nworkers_n/
    │   └── omp_n_njobs_1/
    └── plots/                # from collect_results.py --plot
```

## Prerequisites

- HEWL data: `pyfresean/tests/data/fresean_c_ref/hewl_solution_303K/`
- `FRESEAN_BIN`, `EIGEN_BIN` on PATH; `module load gsl`
- `pip install -e .` from pyfresean root

## One-time CG reference

C inputs:

```bash
bash bench/hewl/prepare_c_covar_inputs.sh
```

pyfresean CG cache + timing:

```bash
python bench/hewl/benchmark.py --mode cg --ncpus 1
```

## Submit sweeps

```bash
bash bench/hewl/submit.sh cg    # py CG reference only
bash bench/hewl/submit.sh c     # C spectral sweep
bash bench/hewl/submit.sh py    # py FRESEAN cases (depends on CG job)
bash bench/hewl/submit.sh all   # all phases
```

## Plot

```bash
python bench/hewl/collect_results.py --case all --plot
python bench/hewl/collect_results.py --case omp1_njobs_n_nworkers_1 --plot
```

## Cases

| Case | OMP | n_jobs | n_workers |
|------|-----|--------|-----------|
| omp1_njobs_n_nworkers_1 | 1 | N | 1 |
| omp1_njobs_n_nworkers_2 | 1 | N | min(2, N) |
| omp1_njobs_n_nworkers_n | 1 | N | N |
| omp_n_njobs_1 | N | 1 | 1 |

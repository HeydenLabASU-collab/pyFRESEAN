# HEWL coarse-grained reference inputs

One-time CG artifacts shared across benchmark sweeps (outside `results/`).

```
cg_reference/
├── c_ref/        # C fresean coarse → topol-cg.mtop, ref-cg.gro, traj-cg.trr
└── pyfresean/    # pyfresean CG cache + bench_t timing JSON
```

Generate C inputs:

```bash
bash bench/hewl/prepare_c_covar_inputs.sh
```

Generate pyfresean CG reference:

```bash
python bench/hewl/benchmark.py --mode cg --ncpus 1
```

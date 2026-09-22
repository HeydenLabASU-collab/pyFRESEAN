# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
The rules for this file:
  * entries are sorted newest-first.
  * summarize sets of changes - don't reproduce every git log comment here.
  * don't ever delete anything.
  * keep the format consistent:
    * do not use tabs but use spaces for formatting
    * 79 char width
    * YYYY-MM-DD date format (following ISO 8601)
  * accompany each entry with github issue/PR number (Issue #xyz)
-->

## [Unreleased]

### Authors
- @amruthesht

### Added
- :meth:`FRESEAN.run` ``compute_corr_matrix`` flag (default ``True``). With
  ``compute_vdos=True`` and ``compute_corr_matrix=False``, total VDOS is
  built from batched diagonal autocorrelation spectra (same lag/window
  pipeline, ``corr_matrix`` phase parallelism) without storing the full
  ``(n_corr, n, n)`` matrix.   ``compute_modes`` requires ``compute_corr_matrix=True``. (PR #10)
- FRESEAN ``benchmark=True`` phase keys ``bench_t_read_traj``,
  ``bench_t_velocity_spectra``, ``bench_t_corr_matrix`` (full or diagonal
  spectral build only), ``bench_t_vdos`` (temperature / ``vdos_norm``
  normalization after a build, plus ``vdos_total`` when VDOS is enabled),
  and ``bench_t_eigen``; ``bench_t_fresean_total`` sums those five. (PR #10)
- Example notebooks ``01_AA-alanine-dipeptide-gas-300K`` (all-atom gas
  FRESEAN) and ``02_AA_CG-hewl-solution-300K`` (AA vs CG HEWL), including
  :meth:`FRESEAN.run` ``read_traj``, ``compute_vdos``, and ``compute_modes``.
  (PR #8)
- GROMACS HEWL solution inputs under ``examples/input_data/MD-HEWL-303K/``
  (topology, MDP, SLURM ``run-slurm.sh``, ``postprocess-prot.sh``).
  (PR #8)
- :meth:`FRESEAN.run` options ``read_traj``, ``compute_vdos``, and
  ``compute_modes`` (each ``True``, ``False``, or a frequency list with
  nearest-bin mapping; defaults ``True``). ``read_traj=False`` reuses
  ``results.velocity_spectra`` / ``results.corr_matrix`` when
  ``results.fresean_frame_key`` matches frame selection and
  ``(n_corr, dt, sigma)``. Reusable intermediates available in
  :attr:`~MDAnalysis.analysis.base.Results`. (PR #7)
- Per-phase parallelism for ``FRESEAN`` via a ``parallel`` dict
  (``velocity_fft``, ``corr_matrix``, ``eigen``), each with ``n_jobs`` and
  ``omp_threads``; ``pyfresean.parallel`` helpers and runtime BLAS limits
  through ``threadpoolctl``. (PR #6)
- Memory-budget upper-triangle tiling for the correlation matrix, optional
  tile ``ThreadPoolExecutor`` when ``corr_matrix.n_jobs > 1``, and
  symmetrization after fill. (PR #6)
- Real FFT path in ``FRESEAN._conclude``: ``rfft`` velocities and lag
  windowing, ``irfft`` for tile cross-correlation; ``float64`` velocity
  storage. (PR #6)
- HEWL benchmark case ``omp_hybrid_vec`` (corr tile pool + BLAS eigen) and
  ``fresean_parallel_for_case()`` mapping; ``submit.sh`` phase
  ``py_fresean_hybrid``. (PR #6)
- ``threadpoolctl>=3.1.0`` dependency and ``pyfresean/tests/test_parallel.py``.
  (PR #6)
- ``benchmark=True`` on ``FRESEAN.run`` and ``CoarseGrain.cg_universe`` /
  ``build_universe`` for wall-time dicts; shared ``bench_t_*`` keys in ``pyfresean.benchmark_keys``. (PR #5)
- HEWL pyfresean vs FRESEAN COARSE (C) benchmark setup under
  ``bench/hewl/``: ``benchmark.py`` modes ``py_cg``,
  ``c_cg``, ``py_fresean``, ``c_spectral``; SLURM ``submit.sh``; and ``collect_results.py`` tables/plots comparing py FRESEAN-only vs C covar+eigen (CG excluded for both). (PR #5)
- ``cg_reference/{pyfresean,c_ref}/`` layout for one-time CG 
  and ``results/`` (generated outputs gitignored). (PR #5)
- Unit tests for FRESEAN and CoarseGrain benchmark timing dicts. (PR #5)
- ``@pytest.mark.c_ref`` regression tests vs stored FRESEAN COARSE (C) for
  alanine dipeptide (gas, 300 K) and HEWL in solution (303 K): CG
  trajectory, eigenvalues, total VDoS, and eigenvectors; optional
  ``--c-ref-plot`` PNG comparisons. (PR #4)
- ``pyfresean.tests.c_ref`` helpers (``io``, ``paths``, ``plotting``,
  per-system modules) and devtools ``generate_fresean_c_ref_*.py`` /
  ``run_c_pipeline.sh`` scripts to regenerate stored C references under
  ``tests/data/fresean_c_ref/``. (PR #4)
- HEWL GROMACS inputs, SLURM/postprocess scripts, and
  C reference data for solution-phase ``c_ref`` (large ``evec`` / MD
  outputs gitignored). (PR #4)
- ``pyfresean.tests.pytutorial_ref`` package; ala dipeptide (gas, 50 K)
  pytutorial reference data under ``tests/data/fresean_pytutorial_ref/``.
  (PR #4)
- ``FRESEAN`` ``n_jobs`` parameter for threaded :meth:`_conclude` work:
  velocity cross-correlation matrix construction (parallel over row index
  ``i``) and per-frequency :func:`numpy.linalg.eigh` diagonalization
  (parallel over frequency bin); ``1`` or ``None`` serial, ``-1`` uses
  :func:`os.cpu_count`. (PR #3)
- MDAnalysis parallel trajectory support for ``FRESEAN`` via
  ``run(n_workers=..., backend="multiprocessing")`` (velocity collection in
  ``_single_frame``; aggregators for ``velocities``, ``freqs``, ``win_time``,
  ``n_dof``, and ``corr_matrix``). (PR #3)
- ``FRESEAN`` ``lag_symmetrization`` parameter: ``"mirror"`` (default,
  FRESEAN_tutorial notebooks) or ``"average"`` (averaging of positive and wrapped negative ``ifft`` lag bins). (PR #2)
- Exact AA reconstruction from CG trajectories using COM-relative displacement
  vectors, with matching velocity backmap and ``track`` / ``ref`` centered modes.
  (PR #1)
- ``reconstruct_aa_universe``, ``aa_universe_from_cg``, and save/load helpers for
  per-frame centered vectors (``aa_rotations.npz``) and ``cg_map.npz``.
  (PR #1)
- Unified coarse-graining I/O: ``aa`` / ``u_cg`` accept a Universe, file path, or
  ``(topology, trajectory)``; ``mapping`` is a ``CoarseGrain`` or
  ``(aa_atomgroup, cg_map)``. (PR #1)
- ``martini`` registered as a supported CG method name (stub; not implemented).
  (PR #1)
- Expanded ``test_coarse_grain.py`` coverage; updated existing
  ``04_CG-alanine-dipeptide-gas-300K`` notebook (backmap roundtrip, FRESEAN);
  added new ``05_CG-hewl`` example. (PR #1)

### Fixed
<!-- Bug fixes -->

### Changed
- FRESEAN work for each :meth:`run` is planned in ``_fresean_run_plan``;
  :meth:`_conclude` and ``benchmark=True`` follow plan flags (trajectory
  read, velocity FFT, full or diagonal correlation spectra, normalization,
  VDOS, modes). Run options are arguments to the planner from
  :meth:`FRESEAN.run`. (PR #10)
- Normalization (and ``vdos_total`` when requested) always runs after a
  fresh full or diagonal spectral build; ``vdos_total`` uses the same
  pre-normalization trace as the temperature / ``vdos_norm`` step. Wall
  time for that step is recorded under ``bench_t_vdos``, not
  ``bench_t_corr_matrix``. (PR #10)
- Repeated ``benchmark=True`` runs on one ``FRESEAN`` instance keep wall
  times for phases skipped by the plan (e.g. ``read_traj=False``); phases
  that run again are reset before timing. (PR #10)
- Benchmark total keys are ``bench_t_cg_total`` (CoarseGrain) and
  ``bench_t_fresean_total`` (FRESEAN). (PR #10)
- HEWL ``collect_results`` FRESEAN breakdown plots aggregate the five
  internal ``bench_t_*`` timings to three curves only: velocity matrix
  (``bench_t_read_traj`` + ``bench_t_velocity_spectra``), corr + norm/VDOS
  (``bench_t_corr_matrix`` + ``bench_t_vdos``), and eigen
  (``bench_t_eigen``). (PR #10)
- Examples revamp: two focused notebooks replace the older MD/CG tutorial
  set; ``examples/README.md`` and ``input_data/`` layout updated. (PR #8)
- ``FRESEAN`` correlation-matrix construction uses vectorized tiles and
  per-phase ``blas_thread_context`` instead of a single global ``n_jobs``.
  (PR #6)
- HEWL ``benchmark.py`` ``py_fresean`` cases reduced to
  ``omp_n_njobs_1_vec``, ``omp1_njobs_n_vec``, and ``omp_hybrid_vec``;
  ``--mode`` is required. (PR #6)
- C CG input generation for HEWL benchmarks consolidated into
  ``benchmark.py --mode c_cg`` (replaces standalone shell script). (PR #5)
- ``c_ref`` FRESEAN comparisons use ``lag_symmetrization="average"`` to match
  FRESEAN COARSE; tutorial notebooks keep ``"mirror"``. (PR #4)
- ``FRESEAN`` default ``n_constraints`` is ``0`` for CG workflows. (PR #1)
- Backmap load, arithmetic, and trajectory write use float32 to match CG and
  centered-vector file storage. (PR #1)

### Deprecated
<!-- Soon-to-be removed features -->

### Removed
- Legacy benchmark key names ``bench_t_spectral``, ``bench_t_total``, and
  ``bench_t_velocity_matrix`` (no aliases in ``benchmark_keys`` or HEWL
  collectors). (PR #10)
- Legacy example notebooks (``01_MD-*`` through ``05_CG-hewl``) and
  ``examples/fresean_metaD_data/``. (PR #8)
- ``FRESEAN`` top-level ``n_jobs`` and legacy ``build_phase_parallelism``
  ``n_jobs`` shorthand; use ``parallel`` instead. (PR #6)
- HEWL benchmark ``all`` mode, mode aliases (``cg`` / ``py`` / ``c``),
  ``--parallel-config`` / ``--n-jobs`` / ``--omp-threads`` CLI overrides,
  and pre-vectorization ``py_fresean`` cases
  (``omp1_njobs_n_nworkers_*``, ``omp_n_njobs_1``). (PR #6)
- ``bench/hewl/prepare_c_covar_inputs.sh`` (superseded by ``c_cg`` mode).
  (PR #5)

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
- ``FRESEAN`` default ``n_constraints`` is ``0`` for CG workflows. (PR #1)
- Backmap load, arithmetic, and trajectory write use float32 to match CG and
  centered-vector file storage. (PR #1)

### Deprecated
<!-- Soon-to-be removed features -->

### Removed
<!-- Removed features -->

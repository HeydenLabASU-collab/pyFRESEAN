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
- Exact AA reconstruction from CG trajectories using COM-relative displacement
  vectors, with matching velocity backmap and ``track`` / ``ref`` centered modes.
- ``reconstruct_aa_universe``, ``aa_universe_from_cg``, and save/load helpers for
  per-frame centered vectors (``aa_rotations.npz``) and ``cg_map.npz``.
- Unified coarse-graining I/O: ``aa`` / ``u_cg`` accept a Universe, file path, or
  ``(topology, trajectory)``; ``mapping`` is a ``CoarseGrain`` or
  ``(aa_atomgroup, cg_map)``.
- ``martini`` registered as a supported CG method name (stub; not implemented).
- Expanded ``test_coarse_grain.py`` coverage; updated existing
  ``04_CG-alanine-dipeptide-gas-300K`` notebook (backmap roundtrip, FRESEAN);
  added new ``05_CG-hewl`` example.

### Fixed
<!-- Bug fixes -->

### Changed
- ``FRESEAN`` default ``n_constraints`` is ``0`` for CG workflows.
- Backmap load, arithmetic, and trajectory write use float32 to match CG and
  centered-vector file storage.

### Deprecated
<!-- Soon-to-be removed features -->

### Removed
<!-- Removed features -->

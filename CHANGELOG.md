# Changelog

All notable repository and software changes are documented here. Dates use
ISO 8601 format.

## Unreleased

- Separate the three formal ReCQC implementations.
- Add reproducible dependency files, repository documentation, and CI checks.
- Define distinct release channels for ReCQC 1.0, 2.0, and Hungarian matching.
- Normalize formatting, remove commented-out debug code, and make exception
  handling explicit across ReCQC 1.0, ReCQC 2.0, and the Hungarian version.
- Align ReCQC 1.0 carbon matching with the original ReCQC 2.0 two-stage
  type-compatible nearest-neighbour workflow while retaining the 1.0 DEPT
  ambiguity rules.
- Correct ReCQC 1.0 maximum-substructure scoring so atom colour/error scores,
  unique cumulative atom unions, and the existing core/environment weighting
  are applied at their intended levels.
- Make ReCQC2.0-speed carbon matching use the CORD-NMR full-score assignment:
  maximize the total green-plus-yellow atom score using the selected global or
  range-specific thresholds, with duplicate peaks retained as separate inputs.
- Isolate malformed SDF records, preserve an existing database when an import
  contains no valid molecules, fix symmetry-aware candidate filtering, and
  make GUI validation/reset/output handling deterministic.

## ReCQC Hungarian — 2026-03-15

- Formal implementation using the new Hungarian-assignment matching algorithm.

## ReCQC 2.0 update — 2026-03-07

- Updated ReCQC 2.0 snapshot.

## ReCQC 2.0 — 2025-11-21

- Published ReCQC 2.0.

## ReCQC 1.0 — 2025-11-21

- Published ReCQC 1.0.

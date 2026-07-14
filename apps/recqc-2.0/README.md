# ReCQC 2.0 — stable release

This directory contains the released and recommended ReCQC 2.0 implementation.
Its scoring behavior should remain stable unless a change is accompanied by a
documented regression dataset and an explicit version update.

## Run

From the repository root:

```powershell
python -m pip install -r requirements.txt
python apps/recqc-2.0/gui.py
```

## Main modules

- `gui.py`: graphical interface and workflow selection.
- `PROCESSSDFFILES.py`: SDF parsing and local database preparation.
- `CarbonScoreProcess.py`: 13C scoring pipeline.
- `HSQCScoreProcess.py`: HSQC scoring pipeline.
- `CombineScorerResult.py`: joint-scoring workflow and results.
- `CarbonScorerResult.py` and `HSQCScorerResult.py`: result presentation.

Generated databases, plots, and result directories are runtime artifacts and
must not be committed.

When publishing results, cite the ReCQC article listed in the repository README
and report the exact release tag or Git commit used.

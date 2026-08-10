# ReCQC 2.0 Speed

This directory contains the optimized ReCQC 2.0 Speed implementation. Carbon
assignment follows the CORD-NMR full-score method: under the selected global or
range-specific thresholds, the one-to-one assignment maximizes the total green
plus yellow atom score. Duplicate experimental shifts remain distinct peak
instances, and symmetry mode groups equal database shifts.

## Run

From the repository root:

```powershell
python -m pip install -r requirements.txt
python ReCQC2.0-speed/gui.py
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

When publishing results, cite the ReCQC article listed in the repository README,
identify the Speed/CORD-NMR full-score matching method, and report the exact
release tag or Git commit used.

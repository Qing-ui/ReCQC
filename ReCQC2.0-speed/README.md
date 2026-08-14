# ReCQC 2.0 Speed

This directory contains the maintained ReCQC 2.0 Speed release. Scientific
background is described in the associated publication.

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

When publishing results, cite the ReCQC article listed in the repository README
and report the exact release tag or Git commit used.

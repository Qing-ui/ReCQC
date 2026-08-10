# ReCQC 1.0 — formal release

This directory contains the maintained ReCQC 1.0 workflow described in the
ReCQC publication. Carbon assignment follows the original ReCQC 2.0 two-stage
matching order: nearest unused type-compatible peaks first, followed by nearest
unused fallback peaks. ReCQC 1.0's ambiguous DEPT labels remain supported.

The maximum-substructure implementation aggregates unique atom scores within
each occurrence, evaluates cumulative atom unions, and then applies the
existing core/environment molecular weighting. No additional chemistry
thresholds are introduced by these maintenance corrections.

## Run

From the repository root:

```powershell
python -m pip install -r apps/recqc-1.0/requirements.txt
python apps/recqc-1.0/gui.py
```

If a robustness fix could alter a 1.0 result, add a regression test and document
the behavior before changing it. Cite the ReCQC article listed in the repository
README when publishing results produced with this version.

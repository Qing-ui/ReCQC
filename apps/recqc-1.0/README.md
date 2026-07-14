# ReCQC 1.0 — formal release

This directory contains the formal ReCQC 1.0 workflow described in the ReCQC
publication. Repository maintenance may improve formatting, diagnostics, and
error handling, while the published scoring and matching behavior remains
unchanged.

## Run

From the repository root:

```powershell
python -m pip install -r apps/recqc-1.0/requirements.txt
python apps/recqc-1.0/gui.py
```

If a robustness fix could alter a 1.0 result, add a regression test and document
the behavior before changing it. Cite the ReCQC article listed in the repository
README when publishing results produced with this version.

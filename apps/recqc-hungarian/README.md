# ReCQC — Hungarian matching implementation

This directory contains the formal ReCQC implementation that uses a new
Hungarian-assignment matching algorithm. It is kept separate from ReCQC 1.0 and
2.0 so that results can be traced to the exact matching method.

## Run

From the repository root:

```powershell
python -m pip install -r requirements.txt
python apps/recqc-hungarian/gui.py
```

## Release policy

- Validate changes against a fixed, documented benchmark.
- Record differences from `apps/recqc-2.0` in `CHANGELOG.md`.
- Publish it under a distinct Hungarian version tag.
- Do not overwrite a ReCQC 1.0 or 2.0 release tag with this implementation.
- Cite the ReCQC article listed in the repository README and report the exact
  Hungarian release tag used.

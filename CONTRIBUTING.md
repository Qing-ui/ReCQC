# Contributing to ReCQC

## Before making a change

1. Open an issue describing the problem, affected version, and expected result.
2. Create a focused branch from `main`.
3. Keep the 1.0, 2.0, and Hungarian behaviors separate; do not move scoring logic
   between versions without an explicit release decision and regression evidence.

## Scientific changes

Changes to parsing, matching, thresholds, assignment, or score aggregation must
include:

- a small redistributable input dataset;
- the expected result before and after the change;
- an explanation of why a changed result is scientifically correct;
- the ReCQC version, Python version, and RDKit version used.

## Code quality

Run these checks before opening a pull request:

```powershell
python -m unittest discover -s tests -v
ruff check apps tests
ruff format --check apps tests
```

Use comments for scientific rationale, assumptions, units, and non-obvious
constraints. Remove commented-out debug code and temporary `print` statements.
Prefer narrow exception types over bare `except:` blocks.

## Pull requests

Keep pull requests small enough to review. State whether the change affects the
stable version, the Hungarian version, or both. Do not commit generated results,
local databases, virtual environments, or IDE files.

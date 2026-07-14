## Summary

Describe the change and why it is needed.

## Affected implementation

- [ ] ReCQC 1.0
- [ ] ReCQC 2.0
- [ ] ReCQC Hungarian
- [ ] Documentation or repository infrastructure only

## Scientific validation

- [ ] No scoring or parsing behavior changed.
- [ ] A fixed input and expected output are included for behavioral changes.
- [ ] Differences between stable and Hungarian results are documented.

## Checks

- [ ] `python -m unittest discover -s tests -v`
- [ ] `ruff check apps tests`
- [ ] `ruff format --check apps tests`
- [ ] Generated data and local databases are not committed.

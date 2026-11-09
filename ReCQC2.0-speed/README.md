# ReCQC 2.0 Speed

ReCQC 2.0 Speed provides faster carbon and HSQC peak matching for the ReCQC 2.0
desktop workflow. Use the same SDF library, peak inputs, and scoring settings
as in [ReCQC 2.0](../apps/recqc-2.0).

ReCQC 2.0 Speed 用于加快 ReCQC 2.0 的碳谱与 HSQC 峰匹配，可使用相同的
SDF 谱库、样品峰表和评分设置。

## Run

From the repository root, install the shared dependencies and launch the app.

```powershell
python -m pip install -r requirements.txt
Set-Location ReCQC2.0-speed
python gui.py
```

This working directory keeps the generated database and results together.

## Matching and scoring

Carbon matching uses the original reference order, float32 differences and
first-occurring tie rule. HSQC uses the original rounded distances, row order
and greedy one-to-one assignment. Carbon typing, exact-coordinate merging,
environment scores, HSQC tolerances and the joint score follow ReCQC 2.0.

The regression suite compares both implementations on duplicate peaks, equal
distances, empty and uneven groups, and typed and untyped matching. A comparison
on 72 molecules from the bundled SDF examples found identical peak pairs,
component scores, and ranked results across 24 configurations. In this dataset,
carbon matching was 1.31–1.39× as fast and HSQC matching with coordinate merging
disabled was 1.50× as fast. These measurements cover matching computation.

Cite the
[ReCQC article](https://doi.org/10.1021/acs.analchem.5c05654) and record the
release tag or commit for published analyses.

ReCQC is distributed under the [repository license](../LICENSE).

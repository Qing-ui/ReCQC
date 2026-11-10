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

## Reference libraries

Select one or more SDF files for each analysis. The first analysis prepares
the library. Later analyses in the same session can reuse it when the file
contents and selection order match. Selecting a different library or editing
an SDF prepares the selected files again. HSQC and joint analyses also reuse
the library's prepared carbon–proton coordinates when available.

SDF paths can contain Chinese characters. Each open result window reads its
own analysis data. The working directory's `chem_data.db` contains the latest
successful analysis.

首次分析会读取并准备所选 SDF。同一会话中，文件内容及选择顺序一致时可复用
已准备的数据。HSQC 和联合分析也可复用已提取的碳氢坐标。换库或修改文件内容后，
程序会重新导入所选文件。

SDF 路径支持中文。已打开的结果窗口会读取各自的分析数据。工作目录中的
`chem_data.db` 保存最近一次成功分析的数据。

## Matching and scoring

Carbon matching uses the original reference order, float32 differences and
first-occurring tie rule. HSQC uses the original rounded distances, row order
and greedy one-to-one assignment. Carbon typing, exact-coordinate merging,
environment scores, HSQC tolerances and the joint score follow ReCQC 2.0.

For larger result lists, plot export adjusts parallelism to the available
processor and memory capacity. Export uses the same plot settings.

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

# ReCQC

ReCQC is a desktop application for dereplication analysis of natural-product
mixtures using 13C NMR and HSQC data. It provides carbon-only, HSQC, and joint
matching workflows through a Tkinter graphical interface.

ReCQC 是一款基于 13C NMR 与 HSQC 数据进行天然产物混合物去重复分析的桌面软件，
提供碳谱匹配、HSQC 匹配和联合匹配三种图形化工作流程。

## Versions / 版本

| Version | Status | Source | Intended use |
| --- | --- | --- | --- |
| ReCQC 1.0 | Released / 正式版 | [`apps/recqc-1.0`](apps/recqc-1.0) | Original 13C NMR dereplication workflow |
| ReCQC 2.0 | Released / 正式版 | [`apps/recqc-2.0`](apps/recqc-2.0) | 13C NMR and HSQC database strategies |
| ReCQC 2.6 | Released / 正式版 | [`ReCQC2.0-speed`](ReCQC2.0-speed) | Performance release |
| ReCQC 2.5 | Released / 正式版 | [`apps/recqc-hungarian`](apps/recqc-hungarian) | Matching release |

All four implementations are maintained as formal ReCQC projects. ReCQC 1.0
and 2.0 are the versions described in the associated publication. Alternative
implementations are kept in separate directories so analyses can record the
exact software version used.

四个实现均作为 ReCQC 正式项目维护。ReCQC 1.0 和 2.0 对应关联论文；其他实现单独保存，
以便分析结果能够明确追溯到所使用的软件版本。

## Repository layout / 目录结构

```text
apps/
  recqc-1.0/          formal ReCQC 1.0 application
  recqc-2.0/          formal ReCQC 2.0 application
  recqc-hungarian/    formal Hungarian matching implementation
docs/                 user documentation
examples/data/        example SDF and spreadsheet data
tests/                 repository-level safety checks
```

## Installation / 安装

ReCQC is currently tested as a Windows desktop application. Python 3.9 is the
compatibility baseline. The recommended installation uses Conda because RDKit
and its graphical dependencies are easier to reproduce that way.

```powershell
conda env create -f environment.yml
conda activate recqc
```

Alternatively, in an existing Python environment:

```powershell
python -m pip install -r requirements.txt
```

Tkinter is part of the standard Windows Python distribution. On Linux, install
the operating system's Tk package separately if it is missing.

## Running ReCQC / 启动程序

ReCQC 1.0：

```powershell
python -m pip install -r apps/recqc-1.0/requirements.txt
python apps/recqc-1.0/gui.py
```

ReCQC 2.0：

```powershell
python apps/recqc-2.0/gui.py
```

ReCQC Hungarian：

```powershell
python apps/recqc-hungarian/gui.py
```

ReCQC 2.0 Speed：

```powershell
python ReCQC2.0-speed/gui.py
```

## Release information / 发布信息

Each directory contains the corresponding maintained release. For scientific
background, refer to the associated publication. Report the exact release tag
or Git commit when publishing results.

各目录包含对应的维护版本。科学背景请参阅相关论文；发表结果时请注明准确的发布标签或
Git 提交。

See [`docs/User Guide.pdf`](docs/User%20Guide.pdf) for the graphical workflow.
Example input files are available in [`examples/data`](examples/data).

## Input and output / 输入与输出

- Input molecular libraries use SDF files.
- NMR shift data must follow the field conventions described in the user guide.
- Runtime databases, plots, and result folders are generated locally and are
  intentionally excluded from version control.
- Keep original input data unchanged and write each analysis to a new output
  directory to preserve reproducibility.

## Development checks / 开发检查

The checks below do not execute scientific calculations or open the GUI. They
verify source syntax, required files, and repository layout.

```powershell
python -m unittest discover -s tests -v
ruff check apps tests
ruff format --check apps tests
```

## Citation / 引用

If you use any ReCQC release, cite the associated article and report the
software version and Git commit used:

> Dong, T. et al. ReCQC: A Novel NMR Data Analysis Platform Leveraging 13C NMR
> and HSQC Database Strategies for Natural Product Dereplication. *Analytical
> Chemistry* **98**(11), 8081-8091 (2026).
> https://doi.org/10.1021/acs.analchem.5c05654

Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff).

### Recommended review / 推荐阅读

For broader context on computer-assisted natural-product dereplication and
structure elucidation, including a discussion of ReCQC among NMR-based
candidate-consistency tools, see:

> Deng, C.; Dong, T.; Zhang, M.; Zi, Y.; Yang, W.; Hao, F.; Zhu, L.; Zhang, S.;
> Li, C.; Ding, Z.; Cai, L. Computer-Assisted Analytical Workflows for Natural
> Product Dereplication, Structure Elucidation, and Bioactivity-Oriented
> Prioritization. *Analytical Chemistry* (2026), ASAP.
> https://doi.org/10.1021/acs.analchem.6c02967

## Contact / 联系方式

For usage questions and reproducible bug reports, contact
[`2638614209@qq.com`](mailto:2638614209@qq.com) or open a GitHub issue that does
not contain confidential research data.

## License / 许可证

ReCQC may be used, copied, modified, and redistributed for non-commercial
scientific research and education, provided that attribution and the license
notice are retained and modified versions are clearly identified. Commercial
use requires prior written permission. See [`LICENSE`](LICENSE).

ReCQC 允许用于非商业科研和教学目的的使用、复制、修改与再分发，但须保留署名及许可声明，
并明确标注修改内容。商业使用须事先获得书面许可，详见 [`LICENSE`](LICENSE)。商业许可请联系
[`2638614209@qq.com`](mailto:2638614209@qq.com)。

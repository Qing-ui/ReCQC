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
| ReCQC (Hungarian) | Released / 正式版 | [`apps/recqc-hungarian`](apps/recqc-hungarian) | New matching algorithm using Hungarian assignment |

All three implementations are maintained as formal ReCQC projects. ReCQC 1.0
and 2.0 are the versions described in the associated publication. The Hungarian
version introduces a new matching algorithm and is kept in a separate directory
so that analyses always record which method produced the results.

三个实现均作为 ReCQC 正式项目维护。ReCQC 1.0 和 2.0 对应关联论文；Hungarian 版本
采用新的匹配算法并单独保存，以便分析结果能够明确追溯到具体方法。

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

See [`CONTRIBUTING.md`](CONTRIBUTING.md) before changing scoring logic. Algorithm
changes should include a fixed input dataset and expected output so that stable
and Hungarian results can be compared explicitly.

## Citation / 引用

If you use ReCQC 1.0, ReCQC 2.0, or the Hungarian implementation, cite the
associated article and report the software version and Git commit used:

> Dong, T. et al. ReCQC: A Novel NMR Data Analysis Platform Leveraging 13C NMR
> and HSQC Database Strategies for Natural Product Dereplication. *Analytical
> Chemistry* **98**(11), 8081-8091 (2026).
> https://doi.org/10.1021/acs.analchem.5c05654

Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## Contact / 联系方式

For usage questions and reproducible bug reports, contact
[`2638614209@qq.com`](mailto:2638614209@qq.com) or open a GitHub issue that does
not contain confidential research data.

## License / 许可证

Copyright is reserved. The repository is publicly visible for scientific
transparency and reproducibility, but copying, modification, redistribution, or
commercial use requires prior written permission. See [`LICENSE`](LICENSE).

本项目保留全部版权。代码公开用于科研透明性与可重复性；复制、修改、再分发或商业使用
须事先取得书面许可，详见 [`LICENSE`](LICENSE)。

# ReCQC 2.6

ReCQC 2.6 provides carbon, HSQC, and joint spectrum matching.

## Run

1. Run `powershell -ExecutionPolicy Bypass -File .\setup_env.ps1` once.
2. Run `powershell -ExecutionPolicy Bypass -File .\run_gui.ps1`.

## Build the Windows executable

After setting up the environment, run:

`powershell -ExecutionPolicy Bypass -File .\build_exe.ps1`

The single-file GUI executable is written to `dist\ReCQC2.6.exe`.
Build output is intentionally excluded from Git; published binaries belong in
GitHub Releases.

The GUI writes `chem_data.db` and result-image directories inside this folder.
No GitHub repository is modified or contacted at runtime.

## License / 许可证

Non-commercial scientific research and educational use, copying, modification,
and redistribution are permitted when attribution and this license notice are
retained and modified versions are clearly identified. Commercial use requires
prior written permission. See `LICENSE`; commercial licensing inquiries may be
sent to `2638614209@qq.com`.

允许将本软件用于非商业科学研究和教育，并可复制、修改和再分发；但须
保留署名及许可声明，并明确标注修改版本及所作更改。商业使用须事先取得
书面许可。完整条款见 `LICENSE`；商业许可请联系 `2638614209@qq.com`。

## Release information

This package contains the maintained ReCQC 2.6 release. See the repository
README and associated publication for general information.

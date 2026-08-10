# ReCQC 2.0 Speed

This is the isolated local performance build of ReCQC 2.0.

## Run

1. Run `powershell -ExecutionPolicy Bypass -File .\setup_env.ps1` once.
2. Run `powershell -ExecutionPolicy Bypass -File .\run_gui.ps1`.

## Build the Windows executable

After setting up the environment, run:

`powershell -ExecutionPolicy Bypass -File .\build_exe.ps1`

The single-file GUI executable is written to `dist\ReCQC2.0-speed.exe`.
Build output is intentionally excluded from Git; published binaries belong in
GitHub Releases.

The GUI writes `chem_data.db` and result-image directories inside this folder.
No GitHub repository is modified or contacted at runtime.

## License

Non-commercial scientific research and educational use, copying, modification,
and redistribution are permitted when attribution and this license notice are
retained and modified versions are clearly identified. Commercial use requires
prior written permission. See `LICENSE`; commercial licensing inquiries may be
sent to `2638614209@qq.com`.

## Compatibility and logic

- Carbon, environment, and HSQC matching/scoring formulas are unchanged.
- SQLite writes are batched and serialized for performance.
- RDKit drawing options are guarded for compatibility across supported versions.
- GUI result plots are generated before their result windows are opened.

## Local verification

- Fresh `acd2015.sdf` run: 2,936 molecules, 81 carbon shifts, and 98 HSQC points.
- All computed fields in the eight result tables matched the unmodified GitHub baseline.
- Fresh end-to-end runs were about 4.3-4.8 times faster; matching/scoring stages were about 6.7 times faster.
- Combined, carbon-only, and HSQC-only result windows were opened successfully.
- 240 generated result PNGs were decoded successfully during GUI smoke testing.

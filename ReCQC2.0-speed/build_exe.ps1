$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "The isolated environment is missing. Run .\setup_env.ps1 first."
}

& $venvPython -m pip install -r requirements-build.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$distPath = Join-Path $PSScriptRoot "dist"
$workPath = Join-Path $PSScriptRoot ".pyinstaller-build"
$specPath = Join-Path $PSScriptRoot ".pyinstaller-spec"

& $venvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name "ReCQC2.0-speed" `
    --distpath $distPath `
    --workpath $workPath `
    --specpath $specPath `
    --hidden-import "matplotlib.backends.backend_tkagg" `
    --hidden-import "rdkit.Chem.Draw.rdMolDraw2D" `
    gui.py

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$exePath = Join-Path $distPath "ReCQC2.0-speed.exe"
if (-not (Test-Path -LiteralPath $exePath)) {
    throw "PyInstaller completed without producing $exePath"
}

Get-Item -LiteralPath $exePath | Select-Object FullName, Length, LastWriteTime

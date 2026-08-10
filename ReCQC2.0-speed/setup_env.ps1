$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPath = Join-Path $PSScriptRoot ".venv"
if (Test-Path -LiteralPath $venvPath) {
    throw "The local .venv already exists; refusing to overwrite it."
}

$pythonCommand = Get-Command py -ErrorAction SilentlyContinue
$codexPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if ($pythonCommand) {
    $basePython = $pythonCommand.Source
} elseif (Test-Path -LiteralPath $codexPython) {
    $basePython = $codexPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand -or $pythonCommand.Source -like "*\WindowsApps\python.exe") {
        throw "A working Python 3.10+ installation was not found."
    }
    $basePython = $pythonCommand.Source
}

& $basePython -m venv $venvPath
$venvPython = Join-Path $venvPath "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Output "Environment ready. Run .\run_gui.ps1 to start ReCQC 2.0 Speed."

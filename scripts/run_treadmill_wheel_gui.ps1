# Launch the treadmill wheel GUI without relying on broken "python" Store aliases.
# Run from Explorer: right-click -> Run with PowerShell, or:  pwsh -File scripts\run_treadmill_wheel_gui.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = $null
foreach ($ver in @("Python312", "Python311", "Python310")) {
    $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\$ver\python.exe"
    if (Test-Path $candidate) {
        $python = $candidate
        break
    }
}
if (-not $python) {
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pyCmd -and $pyCmd.Source -notmatch "WindowsApps") {
        $python = $pyCmd.Source
    }
}
if (-not $python) {
    Write-Host @"
Python not found.

Install (one option):
  winget install Python.Python.3.12

Then either:
  - Run this script again (it prefers Python under %LOCALAPPDATA%\Programs\Python\), or
  - Settings -> Apps -> Advanced app settings -> App execution aliases -> turn OFF python.exe / python3.exe (stops the Microsoft Store stub).

"@
    exit 1
}

Write-Host "Using: $python"
& $python -m src.analysis.treadmill_wheel_gui

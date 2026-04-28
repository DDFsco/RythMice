# Run wheel/treadmill analysis on src/rawdata/test_nosound.dat
# Requires: pip install -r requirements.txt ; Python 3.10+ on PATH as `python`

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
    Write-Error "Python not found. Install from https://www.python.org/downloads/ or run: winget install Python.Python.3.12"
}

$out = Join-Path $root "outputs\wheel_test_nosound"
New-Item -ItemType Directory -Force -Path $out | Out-Null

# Signal sits near ~5.1 V with small steps; hysteresis in volt units (not 0–5 "logic" scale).
& $python -m src.analysis.analyze_treadmill_wheel `
  --input (Join-Path $root "src\rawdata\test_nosound.dat") `
  --outdir $out `
  --time-unit s `
  --threshold-high 5.111 `
  --threshold-low 5.105 `
  --movement-threshold-deg-s 5 `
  --valid-mean-speed-deg-s 5 `
  --smoothing-window-samples 3

Write-Host "Outputs: $out"

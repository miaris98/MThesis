# ==============================================================================
# Serve the local MLflow archive on the external disk.
#
# Companion to tunnel_mlflow.ps1: that one forwards the *live* dashboard off a
# vast.ai instance, this one opens the permanent archive on E:\MThesis_EXP that
# sync_experiments.py fills. Local training runs write straight into the same
# store, so both live here side by side.
#
# Usage:
#   powershell -File mlflow_local.ps1
#   powershell -File mlflow_local.ps1 -Port 10101 -Root "D:\MThesis_EXP"
#   powershell -File mlflow_local.ps1 -Relink        # after a drive-letter change
# ==============================================================================
param(
    [int]$Port = 10100,
    [string]$Root = $env:MTHESIS_EXP_ROOT,
    [string]$Python = "",
    [switch]$Relink,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

if (-not $Root -or $Root -eq "") { $Root = "E:\MThesis_EXP" }
$Store = Join-Path $Root "mlruns"

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "   MLflow archive  |  $Store" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

# --- Locate an interpreter -----------------------------------------------------
# The Windows Store python.exe stub on PATH is not a real interpreter, so prefer
# the project's conda env and only fall back to PATH if it is a working python.
if ($Python -eq "") {
    $candidates = @(
        (Join-Path $env:USERPROFILE "anaconda3\envs\carla_rl\python.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\envs\carla_rl\python.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\python.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $Python = $c; break }
    }
}
if ($Python -eq "") {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $Python = $cmd.Source }
}
if ($Python -eq "" -or -not (Test-Path $Python)) {
    Write-Host "Could not find a Python interpreter. Pass one with -Python <path\to\python.exe>." -ForegroundColor Red
    exit 1
}
Write-Host "Interpreter: $Python" -ForegroundColor DarkGray

# --- Store must exist ----------------------------------------------------------
if (-not (Test-Path $Root)) {
    Write-Host "Experiment root '$Root' not found - is the external disk connected?" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $Store)) {
    Write-Host "No MLflow store yet at $Store - creating an empty one." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $Store | Out-Null
}

# --- Optional path repair ------------------------------------------------------
if ($Relink) {
    Write-Host "`nRepairing artifact paths for the current location..." -ForegroundColor Yellow
    $env:MTHESIS_EXP_ROOT = $Root
    & $Python (Join-Path $PSScriptRoot "sync_experiments.py") --relink
}

# --- MLflow must be installed --------------------------------------------------
# Probe without importing and without touching stderr: redirecting a native exe's
# stderr in Windows PowerShell raises NativeCommandError even on a clean exit, so
# the check would blow up instead of printing the install hint below.
$hasMlflow = & $Python -c "import importlib.util,sys; sys.stdout.write('yes' if importlib.util.find_spec('mlflow') else 'no')"
if ($hasMlflow -ne "yes") {
    Write-Host "`nmlflow is not installed in this interpreter." -ForegroundColor Yellow
    Write-Host "Install it with:" -ForegroundColor Yellow
    Write-Host "    & `"$Python`" -m pip install mlflow" -ForegroundColor Cyan
    exit 1
}

# --- Serve ---------------------------------------------------------------------
# MLflow wants a URI, and a Windows path only becomes a valid one via file:///.
$StoreUri = "file:///" + ($Store -replace '\\', '/')

# MLflow >= 3.16 puts the filesystem backend in maintenance mode and refuses to
# start against it without this opt-out. The archive is a file store by design.
$env:MLFLOW_ALLOW_FILE_STORE = "true"

Write-Host "`nBacking store: $StoreUri" -ForegroundColor Green
Write-Host "Dashboard:     http://localhost:$Port" -ForegroundColor Green
Write-Host "`nPress Ctrl+C to stop.`n" -ForegroundColor DarkGray

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($p)
        Start-Sleep -Seconds 4
        Start-Process "http://localhost:$p"
    } -ArgumentList $Port | Out-Null
}

& $Python -m mlflow ui --backend-store-uri $StoreUri --host 127.0.0.1 --port $Port
